"""
Core class for micromind. Supports helper function for exports. Out-of-the-box
multi-gpu and FP16 training with PyTorch Lightning and much more.

Authors:
    - Francesco Paissan, 2023
    - Gabriele Santini, 2025
"""

from abc import abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
import warnings

import torch
import pytorch_lightning as pl
from torchinfo import summary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import Metric as TorchMetric, MetricCollection
from .utils.metric import CustomMetric

from .utils.helpers import get_logger

logger = get_logger()

# Default configuration
default_cfg = {
    "output_folder": "results",
    "experiment_name": "micromind_exp",
    "opt": "adam",
    "lr": 0.001,
    "debug": False,
}


class MicroMind(pl.LightningModule):
    """
    MicroMind refactored with PyTorch Lightning as backbone.
    
    This is a general-purpose base class that can be extended for any task:
    classification, detection, segmentation, generation, etc.
    
    Key Design Principles:
    - Task-agnostic: No assumptions about data format or task type
    - Modular: Uses modules_dict for flexible network composition
    - Scalable: Built-in multi-GPU, mixed precision, distributed training
    - Extensible: Override only what you need for your specific task
    Arguments
    ---------
        hparams : Optional[Namespace]
            Hyperparameters for the model. Default is None.
    """
    
    def __init__(self, hparams=None):
        super().__init__()
        
        if hparams is None:
            hparams = Namespace(**default_cfg)
        
        # Lightning automatically saves hparams
        if isinstance(hparams, Namespace):
            self.save_hyperparameters(vars(hparams))
        else:
            self.save_hyperparameters(hparams)
        
        # Use 'modules_dict' to avoid conflict with nn.Module.modules()
        self.modules_dict = torch.nn.ModuleDict({})

        # Input shape for export and MAC computation
        self.input_shape = None
        
        # Initialize metric collections for each stage
        self.train_metrics = MetricCollection({}, prefix='train_')
        self.val_metrics = MetricCollection({}, prefix='val_')
        self.test_metrics = MetricCollection({}, prefix='test_')

        # Cache for extraction functions per stage
        self._extraction_fn_cache = {
            'train': None,
            'val': None,
            'test': None,
        }
        
        # TODO decide what to do with this flag
        # Set to False if you want to handle metrics manually in your subclass
        self._auto_compute_metrics = True

        #----------RESOURCES
        self.target_resources = {
            "WM" : None,
            "FLASH" : None,
            "MACCs" : None, # The amount of desired macc (1 to 10 Macc)[0, 1] * 9 + 1 
        }


    @abstractmethod
    def forward(self, batch):
        """
        Forward step of the class. Called during inference and training.
        This method should be overwritten for specific applications.

        Arguments
        ---------
            batch : Any
                Batch as output from the defined DataLoader.
                Could be a tensor, tuple, list, dict - depends on your task.

        Returns
        -------
            pred : Any
                Predictions - completely task-dependent.
                Return whatever makes sense for your task.
        """
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        """
        Computes the loss function for optimization.
        
        This method MUST be implemented.
        It gives you complete control over how loss is computed for your task.
        IMPLEMENTATION CHECKLIST:
        ✓ Return a scalar torch.Tensor 
        ✓ Ensure the tensor requires gradients
        ✓ Handle NaN/Inf cases gracefully
        
        Common mistakes:
        ✗ Returning Python float: return float(loss)  # Wrong!
        ✗ Returning non-scalar: return losses  # Wrong! Use losses.mean()
        ✗ Detaching loss: return loss.detach()  # Wrong! Prevents gradients
        
        Correct example:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, targets)
            return loss  # Already a scalar tensor with gradients

        Arguments
        ---------
            pred : Any
                Output of the forward() function - task-dependent
            batch : Any
                Batch as defined from the DataLoader - task-dependent

        Returns
        -------
            loss : torch.Tensor
                Computed loss value (scalar tensor).
        """
        pass

    def _validated_compute_loss(self, pred, batch) -> torch.Tensor:
        """
        Internal wrapper that validates compute_loss() output.
        """
        loss = self.compute_loss(pred, batch)
        
        # Validation checks
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"compute_loss() must return a torch.Tensor, got {type(loss)}. "
                f"Make sure your loss function returns a tensor."
            )
        
        if loss.dim() != 0:
            raise ValueError(
                f"compute_loss() must return a scalar (0-dim tensor), got shape {loss.shape}. "
                f"Use .mean(), .sum(), or similar to reduce your loss to a scalar."
            )
        
        if not loss.requires_grad:
            warnings.warn(
                "Loss tensor doesn't require gradients. This will prevent training. "
                "Make sure your loss is computed from model parameters.",
                UserWarning
            )
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"compute_loss() returned {loss.item()}. "
                f"Check for numerical instability in your loss computation."
            )
        
        return loss
    
    def add_metric(
        self,
        name: str,
        metric: Union[TorchMetric, Callable],
        stage: str = 'all'
    ):
        """
        Add a metric for tracking during training/validation/testing.
        
        Supports both TorchMetrics instances and custom functions. Custom functions
        are automatically wrapped in CustomMetric for proper state management.
        
        Arguments
        ---------
            name : str
                Name of the metric (will be prefixed with stage name, e.g., 'val_acc')
            metric : Union[TorchMetric, Callable]
                Either:
                - A TorchMetric instance (recommended)
                - A callable function(preds, targets) -> scalar (for custom metrics)
            stage : str
                Which stage to track metric: 'train', 'val', 'test', or 'all'
                Default: 'all'
        
        """
        # Wrap callable functions in CustomMetric
        if callable(metric) and not isinstance(metric, TorchMetric):
            logger.info(
                f"Wrapping custom function '{name}' in CustomMetric. "
                f"Consider using TorchMetrics for better performance."
            )
            metric = CustomMetric(metric)
        
        # Validate it's now a TorchMetric
        if not isinstance(metric, TorchMetric):
            raise TypeError(
                f"Metric must be a TorchMetric or callable, got {type(metric)}"
            )
        
        # Add to appropriate stages
        stages = ['train', 'val', 'test'] if stage == 'all' else [stage]
        
        for s in stages:
            if s not in ['train', 'val', 'test']:
                raise ValueError(f"Invalid stage '{s}'. Must be 'train', 'val', 'test', or 'all'")
            
            metric_collection = getattr(self, f'{s}_metrics')
            
            # Clone metric to avoid sharing state between stages
            metric_collection.add_metrics({name: metric.clone()})
            
            logger.debug(f"Added metric '{name}' to {s} stage")

    def compute_metrics(self, pred, batch, stage: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute metrics for the current batch with automatic format detection.
        
        This method tries to intelligently extract predictions and targets from
        common formats. Override for complex custom behavior.
        
        Supported formats:
        - pred is tuple/list: (predictions, targets) 
        - pred is dict: {'preds': ..., 'targets': ...} or {'logits': ..., 'labels': ...}
        - batch is tuple/list: extracts targets from batch[1]
        
        Arguments
        ---------
            pred : Any
                Output from forward() - task-dependent format
            batch : Any
                Input batch - task-dependent format
            stage : str
                Current stage: 'train', 'val', or 'test'
                
        Returns
        -------
            metrics : Optional[Dict[str, torch.Tensor]]
                Dictionary of computed metrics, or None if no metrics to compute
        """
        # Get the appropriate metric collection
        metric_collection = getattr(self, f'{stage}_metrics')
        
        # If no metrics registered, skip computation
        if len(metric_collection) == 0:
            return None
        
        # Use cached extraction function if available
        if self._extraction_fn_cache[stage] is not None:
            preds, targets = self._extraction_fn_cache[stage](pred, batch)
        else:
            # First time for this stage - detect format and cache
            preds, targets, extraction_fn = self._detect_and_cache_extraction(
                pred, batch, stage
            )
            self._extraction_fn_cache[stage] = extraction_fn
        
        if preds is None or targets is None:
            # Cannot extract predictions/targets automatically
            # User needs to override this method for their specific format
            return None
        
        # Update metrics
        metrics = metric_collection(preds, targets)
        
        return metrics
    
    def _detect_and_cache_extraction(
        self, 
        pred, 
        batch, 
        stage: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Callable]]:
        """
        Detect format once and return both values and extraction function.
        
        This method analyzes the first batch to determine the data format,
        then creates an optimized extraction function for future batches.

        TODO if the format changes mid-training this logic will fail since 
        relies on first batch detection. careful with dict
        
        Returns
        -------
            preds : Optional[torch.Tensor]
                Extracted predictions for this batch
            targets : Optional[torch.Tensor]
                Extracted targets for this batch
            extraction_fn : Optional[Callable]
                Optimized function for future extractions
        """
        # Tuple/List format 
        if isinstance(pred, (tuple, list)) and len(pred) >= 2:
            logger.debug(f"[{stage}] Detected tuple/list format for metrics")
            
            def extract_tuple(p, b):
                return p[0], p[1]
            
            return pred[0], pred[1], extract_tuple
        
        # Dict format 
        elif isinstance(pred, dict):
            pred_key = None
            target_key = None
            
            # Check for prediction keys (in order of preference)
            for key in ['preds', 'predictions', 'logits', 'output']:
                if key in pred and pred[key] is not None:
                    pred_key = key
                    break
            
            # Check for target keys (in order of preference)
            for key in ['targets', 'labels', 'target', 'label']:
                if key in pred and pred[key] is not None:
                    target_key = key
                    break
            
            if pred_key and target_key:
                logger.debug(
                    f"[{stage}] Detected dict format: "
                    f"preds='{pred_key}', targets='{target_key}'"
                )
                
                # Create closure with detected keys
                # Using closure to capture pred_key and target_key
                def make_dict_extractor(pk, tk):
                    def extract_dict(p, b):
                        return p.get(pk), p.get(tk)
                    return extract_dict
                
                extraction_fn = make_dict_extractor(pred_key, target_key)
                return pred[pred_key], pred[target_key], extraction_fn
        
        # Tensor prediction, targets in batch
        elif torch.is_tensor(pred):
            targets = None
            target_source = None
            target_key = None
            
            # Try to extract targets from batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                targets = batch[1]
                target_source = 'batch_tuple'
            elif isinstance(batch, dict):
                for key in ['targets', 'labels', 'target', 'label']:
                    if key in batch and batch[key] is not None:
                        targets = batch[key]
                        target_source = 'batch_dict'
                        target_key = key
                        break
            
            if targets is not None:
                logger.debug(
                    f"[{stage}] Detected tensor format, "
                    f"targets from {target_source}"
                )
                
                if target_source == 'batch_tuple':
                    def extract_tensor_tuple(p, b):
                        return p, b[1]
                    return pred, targets, extract_tensor_tuple
                else:  # batch_dict
                    def make_tensor_dict_extractor(tk):
                        def extract_tensor_dict(p, b):
                            return p, b.get(tk)
                        return extract_tensor_dict
                    
                    extraction_fn = make_tensor_dict_extractor(target_key)
                    return pred, targets, extraction_fn
        
        # Failed to detect format
        logger.warning(
            f"[{stage}] Could not auto-detect prediction/target format. "
            f"Override compute_metrics() in your subclass."
        )
        return None, None, None
   
    def training_step(self, batch, batch_idx):
        """
        Lightning training step - called for each training batch.
        
        This method is task-agnostic and delegates to your implementations
        of forward(), compute_loss(), and compute_metrics().
        """
        pred = self(batch)
        loss = self._validated_compute_loss(pred, batch)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        
        if self._auto_compute_metrics:
            metrics = self.compute_metrics(pred, batch, stage='train')
            if metrics is not None:
                self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step - called for each validation batch.
        
        This method is task-agnostic and delegates to your implementations
        of forward(), compute_loss(), and compute_metrics().
        """
        pred = self(batch)
        loss = self._validated_compute_loss(pred, batch)
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        
        if self._auto_compute_metrics:
            metrics = self.compute_metrics(pred, batch, stage='val')
            if metrics is not None:
                self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """
        Lightning test step - called for each test batch.
        
        This method is task-agnostic and delegates to your implementations
        of forward(), compute_loss(), and compute_metrics().
        """
        pred = self(batch)
        loss = self._validated_compute_loss(pred, batch)
        
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        
        if self._auto_compute_metrics:
            metrics = self.compute_metrics(pred, batch, stage='test')
            if metrics is not None:
                self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and optional learning rate scheduler.
        
        Override this method to customize optimization. By default uses Adam
        with learning rate from hparams.
        TODO this function needs to be extended or removed
        
        Returns
        -------
            optimizer : torch.optim.Optimizer
                Or dict with 'optimizer' and optionally 'lr_scheduler'
        """
        opt_name = getattr(self.hparams, 'opt', 'adam')
        lr = getattr(self.hparams, 'lr', 0.001)
        
        if opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported.")
        
        return optimizer
    
    def set_input_shape(self, input_shape: Tuple):
        """
        Set input shape needed for export and MAC computation.
        
        Arguments
        ---------
            input_shape : Tuple
                Input tensor shape (without batch dimension)
        """
        self.input_shape = input_shape

    def load_checkpoint(
        self, 
        checkpoint_path: Union[Path, str], 
        module_key: Optional[str] = None,
        strip_prefix: Optional[str] = None,
        strict: bool = True,
        map_location: str = "cpu"
    ):
        """
        Flexible checkpoint loader. Can load full Lightning checkpoints, generic PyTorch 
        state_dicts, or specific sub-modules with key remapping.

        Arguments
        ---------
            checkpoint_path : Union[Path, str]
                Path to the checkpoint file.
            module_key : Optional[str]
                If provided, loads the checkpoint ONLY into self.modules_dict[module_key].
                If None, tries to load into the top-level LightningModule.
            strip_prefix : Optional[str]
                If provided, strips this prefix from checkpoint keys before loading.
                Useful when loading a sub-module that was saved as part of a larger model.
            strict : bool
                Whether to strictly enforce that the keys in state_dict match the keys 
                returned by module's state_dict(). Default: True.
            map_location : str
                Device mapping for loading. Default: "cpu".
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        try:
            # Load the file
            loaded_obj = torch.load(checkpoint_path, map_location=map_location)
            
            # 1. Unwrap Lightning Checkpoints or nested dictionaries
            if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
                state_dict = loaded_obj['state_dict']
            elif isinstance(loaded_obj, dict):
                state_dict = loaded_obj
            else:
                # Handle cases where the checkpoint is just the model object (rare but possible)
                state_dict = loaded_obj.state_dict() if hasattr(loaded_obj, "state_dict") else loaded_obj

            # 2. Target a specific module in modules_dict
            if module_key is not None:
                if module_key not in self.modules_dict:
                    raise KeyError(f"Module '{module_key}' not found in modules_dict.")
                
                target_model = self.modules_dict[module_key]
                logger.info(f"Loading weights into specific module: '{module_key}'")
            else:
                target_model = self
                logger.info("Loading weights into full MicroMind model")

            # 3. Handle Prefix Stripping / Key Remapping
            # If we are loading into a specific module, we might need to fix keys.
            final_state_dict = {}
            
            for k, v in state_dict.items():
                new_key = k
                
                # Remove specific prefix if requested (e.g. "backbone.")
                if strip_prefix and new_key.startswith(strip_prefix):
                    new_key = new_key[len(strip_prefix):]
                
                # If loading into a submodule, we often need to remove the wrapper prefix
                # Example: Checkpoint has "modules_dict.backbone.layer1..."
                #          Target (backbone) expects "layer1..."
                if module_key:
                    # Heuristic: if the key starts with the module name or typical wrappers, strip them
                    # Check if key starts with "modules_dict.{module_key}."
                    wrapper_prefix = f"modules_dict.{module_key}."
                    if new_key.startswith(wrapper_prefix):
                        new_key = new_key[len(wrapper_prefix):]
                
                final_state_dict[new_key] = v

            # 4. Load State Dict
            missing, unexpected = target_model.load_state_dict(final_state_dict, strict=strict)
            
            # Logging results
            if len(missing) > 0:
                logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if len(unexpected) > 0:
                logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
                
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise e

    def export(
        self,
        save_dir: Union[Path, str],
        out_format: Optional[str] = "onnx",
        input_shape: Optional[Tuple] = None,
        qbatch: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Export the model to a specified format for deployment.

        Arguments
        ---------
            save_dir : Union[Path, str]
                The directory where the exported model will be saved.
            out_format : Optional[str]
                The format for exporting ('onnx', 'openvino', 'tflite').
            input_shape : Optional[Tuple]
                The input shape of the model.
            qbatch : Optional[torch.Tensor]
                Optional tensor used for PTQ using TFLite.
        """
        from micromind import convert
        
        if qbatch is not None and out_format != "tflite":
            raise AssertionError("Can perform quantization only on TFLite models.")
        
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        
        exp_name = getattr(self.hparams, 'experiment_name', 'micromind_exp')
        save_dir = save_dir.joinpath(exp_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if input_shape is not None:
            self.set_input_shape(input_shape)
        
        assert self.input_shape is not None, "Must specify input_shape for export"
        
        # Put model in eval mode for export
        self.eval()
        
        # TODO check if i have to export self (the Lightning module) or like befor the self.modules_dict
        if out_format == "onnx":
            convert.convert_to_onnx(self, save_dir.joinpath("model.onnx"))
        elif out_format == "openvino":
            convert.convert_to_openvino(self, save_dir)
        elif out_format == "tflite":
            if qbatch is not None:
                qbatch = qbatch.permute(0, 3, 2, 1)
            convert.convert_to_tflite(self, save_dir, batch_quant=qbatch)
        else:
            raise ValueError(f"Unsupported format: {out_format}")
        
        logger.info(f"Model exported to {save_dir} in {out_format} format")

    @torch.no_grad()
    def compute_params(self, log= True) -> Dict[str, int]:
        """
        Compute number of parameters for each module.
        
        Returns
        -------
            params : Dict[str, int]
                Parameter count for each module and total
        """
        was_training = self.training
        self.eval()
        params = {}
        
        for k, m in self.modules_dict.items():
            params[k] = summary(m, verbose=0).total_params
        
        params['total'] = sum(p.numel() for p in self.parameters())
        
        self.train(was_training)
        if log:
            for k, v in params.items():
                logger.info(f"Parameters in {k}: {v}")
            logger.info(f"Total parameters: {params['total']}")

        return params
    
    @torch.no_grad()
    def compute_macs(self, input_shape: Optional[Union[List, Tuple]], log=True) -> Optional[Dict[str, int]]:
        """
        Computes the number of multiply-accumulate operations.
        #TODO add support for dynamic input shapes, and operations
        
        Arguments
        ---------
            input_shape : Union[List, Tuple]
                Input shape for MAC computation.
                
        Returns
        -------
            macs : Dict[str, int]
                MAC count for each module.
        """
        was_training = self.training
        self.eval()
        
        try:
            macs = {}
            last_in = torch.zeros([1] + list(input_shape))
            
            for k, m in self.modules_dict.items():
                macs[k] = summary(m, input_data=last_in, verbose=0).total_mult_adds
                last_in = m(last_in)
                
        except RuntimeError as e:
            warnings.warn(
                f"Could not compute MACs: {e}. This might be due to "
                "dynamic operations. You can estimate this after exporting."
            )
            macs = None

        self.train(was_training)

        if log and macs is not None:
            for k, v in macs.items():
                logger.info(f"MACs in {k}: {v}")
                
        return macs