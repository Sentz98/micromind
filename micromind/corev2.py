"""
Core class for micromind. Supports helper function for exports. Out-of-the-box
multi-gpu and FP16 training with HF Accelerate and much more.

Authors:
    - Francesco Paissan, 2023
    - Gabriele Santini, 2025
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from accelerate import DistributedDataParallelKwargs
from torchinfo import summary

import torch
from accelerate import Accelerator
from tqdm import tqdm
import warnings

from .enum import Stage
from .utils.helpers import get_logger
from .utils.checkpointer import Checkpointer

from .callbacks import (
    TrainingState, CallbackManager, ProgressCallback, ValidationProgressCallback, 
    MetricsCallback, TrainingCallback, EarlyStoppingCallback, TrainingSetupCallback,
    TrainingCleanupCallback, CheckpointingCallback, LearningRateSchedulerCallback
)

logger = get_logger()

# This is used ONLY if you are not using argparse to get the hparams
default_cfg = {
    "output_folder": "results",
    "experiment_name": "micromind_exp",
    "opt": "adam",  # this is ignored if you are overriding the configure_optimizers
    "lr": 0.001,  # this is ignored if you are overriding the configure_optimizers
    "debug": False,
}

class Metric:
    """
    Class for tracking evaluation metrics during training.

    This class allows you to create custom evaluation metrics by providing a
    function to compute the metric and specifying a reduction method.

    Arguments
    ---------
        name : str
            The name of the metric.
        fn : Callable
            A function that computes the metric given predictions and batch data.
        reduction : Optional[str]
            The reduction method for the metric ('sum' or 'mean'). Default is 'mean'.

    Returns
    -------
        Reduced metric. Optionally, you can access the metric history
        before call reduce(clear=True) : torch.Tensor

    Example
    -------
    .. doctest::

        >>> from micromind import Metric, Stage
        >>> import torch

        >>> def custom_metric(pred, batch):
        ...     # Replace this with your custom metric calculation
        ...     return pred - batch

        >>> metric = Metric("Custom Metric", custom_metric, reduction="mean")
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> batch = torch.tensor([0.5, 1.5, 2.5])
        >>> metric(pred, batch, stage=Stage.train)
        >>> metric.history
        {0: [tensor([0.5000, 0.5000, 0.5000])], 1: [], 2: []}
        >>> metric.reduce(Stage.train)
        0.5
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        reduction: Optional[str] = "mean",
        eval_only: Optional[bool] = False,
        eval_period: Optional[int] = 1,
    ):
        self.name = name
        self.fn = fn
        self.reduction = reduction
        self.eval_only = eval_only
        self.eval_period = eval_period

        self.history = {s: [] for s in [Stage.train, Stage.val, Stage.test]}

    def __call__(self, pred, batch, stage, device="cpu"):
        dat = self.fn(pred, batch)
        if dat.ndim == 0:
            dat = dat.unsqueeze(0)

        self.history[stage].append(dat)

    def reduce(self, stage, clear=False):
        """
        Compute and return the metric for a given prediction and batch data.

        Arguments
        ---------
            pred : torch.Tensor
                The model's prediction.
            batch : torch.Tensor
                The ground truth or target values.
            stage : Stage
                The current stage (e.g., Stage.train).
            device Optional[str]
                The device on which to perform the computation. Default is 'cpu'.
        """
        # Fix: Check if history is empty to avoid runtime error
        if not self.history[stage]:
            logger.warning(f"No data for metric {self.name} in stage {stage}")
            return 0.0

        if self.reduction == "mean":
            tmp = torch.cat(self.history[stage], dim=0).mean()
        elif self.reduction == "sum":
            tmp = torch.cat(self.history[stage], dim=0).sum()
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

        if clear:
            self.history[stage] = []

        return tmp.item()


class MicroMind(ABC):
    """
    MicroMind is an abstract base class for creating and training deep learning
    models. Handles training on multi-gpu via accelerate (using DDP and other
    distributed training strategies). It automatically handles the device
    management for the training and the micromind's export capabilities to onnx,
    OpenVino and TFLite.

    Arguments
    ---------
        hparams : Optional[Namespace]
            Hyperparameters for the model. Default is None.

    """

    def __init__(self, hparams=None, disable_progress=False):
        if hparams is None:
            hparams = Namespace(**default_cfg)

        # Core attributes
        self.modules = torch.nn.ModuleDict({})
        self.hparams = hparams
        self.input_shape = None
        self.current_epoch = 0
        self.start_epoch = 0  # Fix: Initialize start_epoch
        self.disable_progress = disable_progress

        # Fix: Initialize these attributes that are used in init_devices
        self.datasets = {}
        self.metrics = []
        self.checkpointer = None
        self.debug = False

        # Device management
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])  # Fix: Pass as list
        self.device = self.accelerator.device

        # Callback system
        self.callback_manager = CallbackManager()
        self._setup_default_callbacks()
    
    def _setup_default_callbacks(self):
        """Setup default callbacks for backward compatibility"""
        # Core training callbacks
        self.callback_manager.add_callback(TrainingSetupCallback())
        self.callback_manager.add_callback(TrainingCleanupCallback())
        self.callback_manager.add_callback(CheckpointingCallback())
        
        # Progress and metrics callbacks
        self.callback_manager.add_callback(ProgressCallback(disable_progress=self.disable_progress))
        self.callback_manager.add_callback(ValidationProgressCallback(disable_progress=self.disable_progress))
        self.callback_manager.add_callback(MetricsCallback())
        
        # Learning rate scheduler callback (defaults to batch-level stepping)
        self.callback_manager.add_callback(LearningRateSchedulerCallback(step_on='batch'))

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a custom callback to the training loop"""
        self.callback_manager.add_callback(callback)
        logger.info(f"Added callback: {callback.__class__.__name__}")

    def remove_callback(self, callback: TrainingCallback) -> None:
        """Remove a callback from the training loop"""
        self.callback_manager.remove_callback(callback)
        logger.info(f"Removed callback: {callback.__class__.__name__}")

    def remove_callback_by_type(self, callback_type: type) -> None:
        """Remove all callbacks of a specific type"""
        callbacks_to_remove = [cb for cb in self.callback_manager.callbacks if isinstance(cb, callback_type)]
        for callback in callbacks_to_remove:
            self.callback_manager.remove_callback(callback)
            logger.info(f"Removed callback by type: {callback.__class__.__name__}")

    def log_active_callbacks(self) -> None:
        """Log all currently active callbacks"""
        logger.info("=== Active Callbacks ===")
        for i, callback in enumerate(self.callback_manager.callbacks):
            logger.info(f"{i+1}. {callback.__class__.__name__}")
        logger.info("========================")

    @abstractmethod
    def forward(self, batch):
        """
        Forward step of the class. It gets called during inference and optimization.
        This method should be overwritten for specific applications.

        Arguments
        ---------
            batch : torch.Tensor
                Batch as output from the defined DataLoader.

        Returns
        -------
            pred : Union[torch.Tensor, Tuple]
                Predictions - this depends on the task.
        """
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        """
        Computes the cost function for the optimization process.  It return a
        tensor on which backward() is called. This method should be overwritten
        for the specific application.

        Arguments
        ---------
            pred : Union[torch.Tensor, Tuple]
                Output of the forward() function
            batch : torch.Tensor
                Batch as defined from the DataLoader.

        Returns
        -------
            loss : torch.Tensor
                Compute cost function.
        """
        pass

    def set_input_shape(self, input_shape: Tuple = (3, 224, 224)):
        """Setter function for input_shape.

        Arguments
        ---------
        input_shape : Tuple
            Input shape of the forward step.

        """
        self.input_shape = input_shape
        self.modules.input_shape = input_shape

    def load_modules(self, checkpoint_path: Union[Path, str]):
        """Loads models for path.

        Arguments
        ---------
        checkpoint_path : Union[Path, str]
            Path to the checkpoint where the modules are stored.

        """
        dat = torch.load(checkpoint_path, map_location="cpu")

        modules_keys = list(self.modules.keys())
        for k in self.modules:
            try:
                self.modules[k].load_state_dict(dat[k])
            except Exception as e:  # maybe saved with DDP
                tmp = f""" There was a problem loading the checkpoint...
                    Maybe trained with DDP... trying to load it anyways.
                    Error was {type(e).__name__}.
                    """
                warnings.warn(" ".join(tmp.split()))

                self.modules[k] = torch.nn.DataParallel(self.modules[k])
                self.modules[k].load_state_dict(dat[k])
                self.modules[k] = self.modules[k].module

            logger.info("Successfully loaded model from checkpoint.")

            modules_keys.remove(k)

        if len(modules_keys) != 0:
            logger.info(f"Couldn't find a state_dict for modules {modules_keys}.")

    def export(
        self,
        save_dir: Union[Path, str],
        out_format: Optional[str] = "onnx",
        input_shape: Optional[Tuple] = None,  # Fix: Changed type hint
        qbatch: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Export the model to a specified format for deployment.
        TFLite and OpenVINO need a Linux machine to be exported.


        Arguments
        ---------
        save_dir : Union[Path, str]
            The directory where the exported model will be saved.
        out_format : Optional[str]
            The format for exporting the model. Default is 'onnx'.
        input_shape : Optional[Tuple]
            The input shape of the model. If not provided, the input shape
            specified during model creation is used.
        qbatch : Optional[torch.Tensor]
            Optional tensor used for PTQ using TFLite. Channels dimension is
            permuted automatically.

        """
        from micromind import convert

        if qbatch is not None:
            if out_format != "tflite":
                raise AssertionError("Can perform quantization only on TFLite models.")
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir = save_dir.joinpath(self.hparams.experiment_name)

        # Fix: Use input_shape parameter if provided
        if input_shape is not None:
            self.set_input_shape(input_shape)
            
        assert (
            self.input_shape is not None
        ), "You should pass the input_shape of the model."
        self.add_forward_to_modules()

        if out_format == "onnx":
            convert.convert_to_onnx(self.modules, save_dir.joinpath("model.onnx"))
        elif out_format == "openvino":
            convert.convert_to_openvino(self.modules, save_dir)
        elif out_format == "tflite":
            qbatch = qbatch.permute(0, 3, 2, 1)
            convert.convert_to_tflite(self.modules, save_dir, batch_quant=qbatch)

    def configure_optimizers(self):
        """Configures and defines the optimizer for the task. Defaults to adam
        with lr=0.001; It can be overwritten by either passing arguments from the
        command line, or by overwriting this entire method.
        Scheduler step is called every optimization step.

        Returns
        -------
        Optimizer and learning rate scheduler.
            : Union[Tuple[torch.optim.Adam, None], torch.optim.Adam]

        """
        assert self.hparams.opt in [
            "adam",
            "sgd",
        ], f"Optimizer {self.hparams.opt} not supported."
        if self.hparams.opt == "adam":
            opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(self.modules.parameters(), self.hparams.lr)

        return opt

    def __call__(self, *x, **xv):
        """Just forwards everything to the forward method."""
        return self.forward(*x, **xv)

    def add_forward_to_modules(self):
        """Exports MicroMind forward function to its core ModuleList."""
        bound_method = self.forward.__get__(self.modules, self.modules.__class__)
        setattr(self.modules, "forward", bound_method)
        self.modules.device = self.device

    @torch.no_grad()
    def compute_params(self):
        """Computes the number of parameters for the modules inside `self.modules`.
        Returns a dictionary with the parameter count for each module.

        Returns
        -------
        Parameter count for self.modules. : Dict[int]
        """
        self.eval()
        params = {}
        for k, m in self.modules.items():
            params[k] = summary(m, verbose=0).total_params

        return params

    @torch.no_grad()
    def compute_macs(self, input_shape: Union[List, Tuple]):
        """Computes the number of multiply-add for the modules inside `self.modules`.
        Returns a dictionary with the MAC count for each module.

        Arguments
        ---------
        input_shape : Union[List, Tuple]
            Needed for MAC computation.

        Returns
        -------
        MAC count for self.modules. : Dict[int]
        """
        self.eval()

        try:
            macs = {}
            last_in = torch.zeros([1] + list(input_shape))
            for k, m in self.modules.items():
                macs[k] = summary(m, input_data=last_in, verbose=0).total_mult_adds
                last_in = m(last_in)
        except RuntimeError:
            tmp = """
            Could not compute the number of MACs of your MicroMind. Might be due
            to on-the-fly data augmentation or something similar. You can, however,
            estimate this more accurately after exporting the model.
            """
            warnings.warn(" ".join(tmp.split()))
            macs = None

        return macs

    def init_devices(self):
        """Initializes the data pipeline and modules for DDP and accelerated inference.
        To control the device selection, use `accelerate config`."""

        # Initialize optimizer if not already done
        if not hasattr(self, "opt"):
            self.opt = self.configure_optimizers()

        # pass each module through DDP independently
        convert = list(self.modules.values())
        if hasattr(self, "opt"):
            convert += [self.opt]

        if hasattr(self, "lr_sched"):
            convert += [self.lr_sched]

        if hasattr(self, "datasets") and self.datasets:
            # if the datasets are store here, prepare them for DDP
            convert += list(self.datasets.values())

        accelerated = self.accelerator.prepare(*convert)
        for idx, key in enumerate(self.modules):
            self.modules[key] = accelerated[idx]
        self.accelerator.register_for_checkpointing(self.modules)

        if hasattr(self, "opt"):
            self.opt = accelerated[len(self.modules)]
            self.accelerator.register_for_checkpointing(self.opt)

        if hasattr(self, "lr_sched"):
            self.lr_sched = accelerated[1 + len(self.modules)]
            self.accelerator.register_for_checkpointing(self.lr_sched)

        if hasattr(self, "datasets") and self.datasets:
            for i, key in enumerate(list(self.datasets.keys())[::-1]):
                self.datasets[key] = accelerated[-(i + 1)]

        self.modules.to(self.device)

    def eval(self):
        self.modules.eval()

    def train(
        self,
        epochs: int = 1,
        datasets: Dict = {},
        metrics: List[Metric] = [],
        checkpointer: Optional[Checkpointer] = None,
        debug: Optional[bool] = False,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> None:
        """
        Enhanced training method with callback support and detailed logging.
        
        Arguments
        ---------
        epochs : int
            The number of training epochs.
        datasets : Dict
            A dictionary of dataset loaders.
        metrics : Optional[List[Metric]]
            A list of metrics to track during training.
        checkpointer : Optional[Checkpointer]
            Checkpointer for saving model state.
        debug : bool
            Whether to run in debug mode.
        callbacks : Optional[List[TrainingCallback]]
            Additional callbacks to add for this training session.
        """
        logger.info("=== STARTING TRAINING ===")
        logger.info(f"Epochs: {epochs}, Debug: {debug}")
        
        # Setup
        self.datasets = datasets
        self.metrics = metrics
        self.checkpointer = checkpointer
        self.debug = debug
        
        # Add temporary callbacks for this training session
        temp_callbacks = []
        if callbacks:
            logger.info("Adding temporary callbacks for this training session:")
            for callback in callbacks:
                self.add_callback(callback)
                temp_callbacks.append(callback)

        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        # Log all active callbacks before training
        self.log_active_callbacks()

        # Initialize training state
        total_batches = len(self.datasets["train"])
        state = TrainingState(
            epoch=0,
            batch_idx=0,
            total_epochs=epochs,
            total_batches=total_batches
        )

        # Call training start callbacks (this handles setup)
        logger.info("🚀 Executing on_train_start callbacks...")
        self.callback_manager.on_train_start(self, state)

        try:
            # Check for early stopping callback
            early_stopping = None
            for callback in self.callback_manager.callbacks:
                if isinstance(callback, EarlyStoppingCallback):
                    early_stopping = callback
                    logger.info(f"Early stopping callback found: {callback.__class__.__name__}")
                    break

            for e in range(self.start_epoch + 1, epochs + 1):
                logger.info(f"📊 Starting epoch {e}/{epochs}")
                self.current_epoch = e
                state.epoch = e
                
                # Epoch start callbacks
                logger.info(f"⚡ Executing on_epoch_start callbacks for epoch {e}...")
                self.callback_manager.on_epoch_start(self, state)
                
                # Training epoch
                logger.info(f"🎓 Training epoch {e}...")
                epoch_metrics = self._train_epoch(state)
                
                # Validation if available
                if "val" in datasets:
                    logger.info(f"🔍 Validating epoch {e}...")
                    val_metrics = self._validate_epoch(state)
                    epoch_metrics.update(val_metrics)
                
                state.metrics = epoch_metrics
                
                # Log epoch metrics
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                logger.info(f"📈 Epoch {e} metrics: {metrics_str}")
                
                # Epoch end callbacks (this handles checkpointing)
                logger.info(f"✅ Executing on_epoch_end callbacks for epoch {e}...")
                self.callback_manager.on_epoch_end(self, state)
                
                # Check early stopping
                if early_stopping and early_stopping.should_stop_training():
                    logger.info(f"⏹️ Early stopping triggered at epoch {e}")
                    break
                
                if e >= 1 and self.debug:
                    logger.info("🐛 Debug mode: stopping after first epoch")
                    break

        except Exception as e:
            logger.error(f"❌ Training failed with error: {str(e)}")
            raise
        finally:
            # Training end callbacks (this handles cleanup)
            logger.info("🏁 Executing on_train_end callbacks...")
            self.callback_manager.on_train_end(self, state)
            
            # Remove temporary callbacks
            if temp_callbacks:
                logger.info("🧹 Removing temporary callbacks...")
                for callback in temp_callbacks:
                    self.remove_callback(callback)
                    
            logger.info("=== TRAINING COMPLETED ===")

    def _train_epoch(self, state: TrainingState) -> Dict[str, float]:
        """Execute one training epoch with callbacks"""
        self.modules.train()
        loss_epoch = 0
        
        for idx, batch in enumerate(self.datasets["train"]):
            state.batch_idx = idx
            state.batch = batch
            state.stage = Stage.train
            
            # Batch start callbacks
            if idx == 0:  # Log only for first batch to avoid spam
                logger.debug("⏰ Executing on_batch_start callbacks...")
            self.callback_manager.on_batch_start(self, state)
            
            # Prepare batch
            if isinstance(batch, list):
                batch = [b.to(self.device) for b in batch]
            
            self.opt.zero_grad()
            
            # Forward pass
            with self.accelerator.autocast():
                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                loss_epoch += loss.item()
            
            state.outputs = model_out
            state.loss = loss
            
            # Loss computed callback
            if idx == 0:
                logger.debug("💰 Executing on_loss_computed callbacks...")
            self.callback_manager.on_loss_computed(self, state)
            
            # Backward pass
            self.accelerator.backward(loss)
            self.opt.step()
            
            # Backward end callback
            if idx == 0:
                logger.debug("⬅️ Executing on_backward_end callbacks...")
            self.callback_manager.on_backward_end(self, state)
            
            # Batch end callbacks (includes metrics computation and progress updates)
            if idx == 0:
                logger.debug("🔚 Executing on_batch_end callbacks...")
            self.callback_manager.on_batch_end(self, state)
            
            if self.debug and idx > 10:
                logger.info("🐛 Debug mode: stopping after 10 batches")
                break
        
        # Compute final training metrics
        train_metrics = {}
        for m in self.metrics:
            if (state.epoch) % m.eval_period == 0 and not m.eval_only:
                train_metrics["train_" + m.name] = m.reduce(Stage.train, True)
        
        train_metrics.update({"train_loss": loss_epoch / (state.batch_idx + 1)})
        
        return train_metrics

    @torch.no_grad()
    def _validate_epoch(self, state: TrainingState) -> Dict[str, float]:
        """Execute validation epoch with callbacks"""
        assert "val" in self.datasets, "Validation dataloader was not specified."
        
        self.modules.eval()
        loss_epoch = 0
        
        # Validation start callback
        logger.debug("🔍 Executing on_validation_start callbacks...")
        self.callback_manager.on_validation_start(self, state)
        
        with self.accelerator.autocast():
            for idx, batch in enumerate(self.datasets["val"]):
                state.batch_idx = idx
                state.batch = batch
                state.stage = Stage.val
                
                if isinstance(batch, list):
                    batch = [b.to(self.device) for b in batch]
                
                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                
                state.outputs = model_out
                state.loss = loss
                
                # Batch end callback for validation (includes metrics and progress)
                if idx == 0:
                    logger.debug("🔚 Executing validation on_batch_end callbacks...")
                self.callback_manager.on_batch_end(self, state)
                
                loss_epoch += loss.item()
                
                if self.debug and idx > 10:
                    logger.info("🐛 Debug mode: stopping validation after 10 batches")
                    break
        
        # Compute validation metrics
        val_metrics = {}
        for m in self.metrics:
            if (state.epoch) % m.eval_period == 0:
                val_metrics["val_" + m.name] = m.reduce(Stage.val, True)
        
        val_metrics.update({"val_loss": loss_epoch / (state.batch_idx + 1)})
        
        # Validation end callback
        logger.debug("✅ Executing on_validation_end callbacks...")
        self.callback_manager.on_validation_end(self, state)
        
        return val_metrics

    @torch.no_grad()
    def validate(self) -> Dict:
        """Legacy validation method for backward compatibility"""
        if "val" not in self.datasets:
            logger.warning("No validation dataset available")
            return {}
            
        state = TrainingState(
            epoch=self.current_epoch,
            batch_idx=0,
            total_epochs=1,
            total_batches=len(self.datasets["val"])
        )
        return self._validate_epoch(state)

    @torch.no_grad()
    def test(self, datasets: Dict = {}, metrics: List[Metric] = []) -> Dict:
        """Test method with callback support"""
        assert "test" in datasets, "Test dataloader was not specified."
        logger.info("🧪 Starting testing...")
        
        self.modules.eval()

        total_batches = len(datasets["test"])
        state = TrainingState(
            epoch=0,
            batch_idx=0,
            total_epochs=1,
            total_batches=total_batches,
            stage=Stage.test
        )

        loss_epoch = 0
        
        # Create a simple progress bar callback for testing
        if self.accelerator.is_local_main_process:
            pbar = tqdm(datasets["test"], unit="batches", ascii=True, 
                       dynamic_ncols=True, desc="Testing...")
        
        with self.accelerator.autocast():
            for idx, batch in enumerate(datasets["test"]):
                state.batch_idx = idx
                state.batch = batch
                
                if isinstance(batch, list):
                    batch = [b.to(self.device) for b in batch]

                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                
                for m in metrics:
                    m(model_out, batch, Stage.test, self.device)

                loss_epoch += loss.item()
                
                if self.accelerator.is_local_main_process:
                    pbar.set_postfix(loss=loss_epoch / (idx + 1))

        if self.accelerator.is_local_main_process:
            pbar.close()

        test_metrics = {"test_" + m.name: m.reduce(Stage.test, True) for m in metrics}
        test_metrics.update({"test_loss": loss_epoch / (idx + 1)})
        
        s_out = ("Testing " + 
                " - ".join([f"{k}: {v:.2f}" for k, v in test_metrics.items()]) + 
                "; ")
        logger.info(s_out)

        return test_metrics