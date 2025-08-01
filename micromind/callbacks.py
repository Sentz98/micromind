"""
Callback class for micromind core with enhanced training setup callbacks.

Authors:
    - Gabriele Santini, 2025
"""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import warnings

import torch

from .enum import Stage
from .utils.helpers import get_logger

logger = get_logger()

    
@dataclass
class TrainingState:
    """Container for training state information passed to callbacks"""
    epoch: int
    batch_idx: int
    total_epochs: int
    total_batches: int
    loss: Optional[torch.Tensor] = None
    outputs: Optional[Any] = None
    batch: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    stage: Optional[int] = None
    
    def get_progress(self) -> Dict[str, float]:
        """Get training progress as percentages"""
        return {
            'epoch_progress': (self.epoch - 1) / self.total_epochs if self.total_epochs > 0 else 0.0,
            'batch_progress': self.batch_idx / self.total_batches if self.total_batches > 0 else 0.0
        }


class TrainingCallback(ABC):
    """
    Abstract base class for training callbacks.
    
    Callbacks allow you to hook into different points of the training process
    and add custom functionality without modifying the core training loop.
    """
    
    def on_train_start(self, trainer, state: TrainingState) -> None:
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer, state: TrainingState) -> None:
        """Called at the end of training"""
        pass
    
    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        """Called at the end of each epoch"""
        pass
    
    def on_batch_start(self, trainer, state: TrainingState) -> None:
        """Called at the beginning of each batch"""
        pass
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        """Called at the end of each batch"""
        pass
    
    def on_validation_start(self, trainer, state: TrainingState) -> None:
        """Called at the beginning of validation"""
        pass
    
    def on_validation_end(self, trainer, state: TrainingState) -> None:
        """Called at the end of validation"""
        pass
    
    def on_loss_computed(self, trainer, state: TrainingState) -> None:
        """Called after loss computation"""
        pass
    
    def on_backward_end(self, trainer, state: TrainingState) -> None:
        """Called after backward pass"""
        pass


class CallbackManager:
    """
    Class for manage and execute callbacks during training

    Example
    -------
    .. doctest::

        >>> trainer = MicroMind(hparams)
        >>> ## Remove default LR scheduler if wanted a personalized one
        >>> trainer.remove_callback_by_type(LearningRateSchedulerCallback)
        >>> # Add custom callback
        >>> trainer.add_callback(EarlyStoppingCallback(patience=15))
        >>> trainer.add_callback(GradientClippingCallback(max_norm=1.0))
        >>> trainer.add_callback(WarmupCallback(warmup_epochs=10))
        >>> # Training
        >>> trainer.train(epochs=100, datasets=datasets, metrics=metrics)
    """

    
    def __init__(self, callbacks: List[TrainingCallback] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback to the manager"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: TrainingCallback) -> None:
        """Remove a callback from the manager"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _call_callbacks(self, method_name: str, trainer, state: TrainingState) -> None:
        """Execute a specific callback method on all callbacks"""
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name, None)
                if method is not None:
                    method(trainer, state)
            except Exception as e:
                logger.warning(f"Callback {callback.__class__.__name__}.{method_name} failed: {e}")
    
    def on_train_start(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_train_start', trainer, state)
    
    def on_train_end(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_train_end', trainer, state)
    
    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_epoch_start', trainer, state)
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_epoch_end', trainer, state)
    
    def on_batch_start(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_batch_start', trainer, state)
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_batch_end', trainer, state)
    
    def on_validation_start(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_validation_start', trainer, state)
    
    def on_validation_end(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_validation_end', trainer, state)
    
    def on_loss_computed(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_loss_computed', trainer, state)
    
    def on_backward_end(self, trainer, state: TrainingState) -> None:
        self._call_callbacks('on_backward_end', trainer, state)

#--------------------------------------------------------------------------------------------
# Built-in callbacks for common functionality
#--------------------------------------------------------------------------------------------

class TrainingSetupCallback(TrainingCallback):
    """
    Callback that handles the training setup logic (formerly in on_train_start).
    This includes optimizer configuration, device initialization, and checkpoint recovery.
    """
    
    def on_train_start(self, trainer, state: TrainingState) -> None:
        """Initialize training setup"""
        # Pass debug status to checkpointer
        if trainer.checkpointer is not None:
            trainer.checkpointer.debug = trainer.hparams.debug

        # Configure optimizers
        init_opt = trainer.configure_optimizers()
        if isinstance(init_opt, (list, tuple)):
            trainer.opt, trainer.lr_sched = init_opt
        else:
            trainer.opt = init_opt

        # Initialize devices
        trainer.init_devices()

        # Handle checkpoint recovery
        trainer.start_epoch = 0
        if trainer.checkpointer is not None:
            ckpt = trainer.checkpointer.recover_state()
            if ckpt is not None:
                accelerate_path, trainer.start_epoch = ckpt
                trainer.accelerator.load_state(accelerate_path)
                logger.info(f"Recovered checkpoint from epoch {trainer.start_epoch}")
        else:
            tmp = """
                You are not passing a checkpointer to the training function, 
                thus no status will be saved. If this is not the intended behaviour 
                please check https://micromind-toolkit.github.io/docs/.
            """
            warnings.warn(" ".join(tmp.split()))

        # Log training start information
        if trainer.accelerator.is_local_main_process:
            logger.info(f"Starting from epoch {trainer.start_epoch + 1}. "
                       f"Training is scheduled for {state.total_epochs} epochs.")


class TrainingCleanupCallback(TrainingCallback):
    """
    Callback that handles training cleanup logic (formerly in on_train_end).
    Can be extended for custom cleanup operations.
    """
    
    def on_train_end(self, trainer, state: TrainingState) -> None:
        """Clean up after training completion"""
        # Base cleanup - can be extended by subclasses
        if trainer.accelerator.is_local_main_process:
            logger.info("Training completed successfully.")
        
        # Clear any temporary state if needed
        # This is where you could add model saving, final logging, etc.
        pass


class CheckpointingCallback(TrainingCallback):
    """
    Callback that handles model checkpointing during training.
    Separated from the main training loop for better modularity.
    """
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        """Handle checkpointing at the end of each epoch"""
        if (hasattr(trainer, 'datasets') and "val" in trainer.datasets and 
            trainer.accelerator.is_local_main_process and 
            trainer.checkpointer is not None and
            state.metrics is not None):
            
            # Split metrics into train and validation
            train_metrics = {k: v for k, v in state.metrics.items() if k.startswith('train_')}
            val_metrics = {k: v for k, v in state.metrics.items() if k.startswith('val_')}
            
            # Call checkpointer
            trainer.checkpointer(trainer, train_metrics, val_metrics)


class ProgressCallback(TrainingCallback):
    """Built-in callback for progress bar management"""
    
    def __init__(self, disable_progress: bool = False):
        self.pbar = None
        self.epoch_loss = 0
        self.running_metrics = {}
        self.disable_progress = disable_progress
    
    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        if not self.disable_progress and state.stage != Stage.val:
            self.pbar = tqdm(
                total=state.total_batches,
                unit="batches",
                ascii=True,
                dynamic_ncols=True,
                disable=not trainer.accelerator.is_local_main_process,
            )
            self.pbar.set_description(f"Running epoch {state.epoch}/{state.total_epochs}")
            self.epoch_loss = 0
            self.running_metrics = {}
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if self.pbar is not None and state.loss is not None and state.stage == Stage.train:
            self.epoch_loss += state.loss.item()
            
            # Update running metrics
            for m in trainer.metrics:
                if ((state.epoch) % m.eval_period == 0 and not m.eval_only and 
                    len(m.history[Stage.train]) > 0):
                    self.running_metrics["train_" + m.name] = m.reduce(Stage.train)
            
            self.running_metrics.update({"train_loss": self.epoch_loss / (state.batch_idx + 1)})
            self.pbar.set_postfix(**self.running_metrics)
            self.pbar.update(1)
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        if self.pbar is not None and state.stage != Stage.val:
            self.pbar.close()
            self.pbar = None


class ValidationProgressCallback(TrainingCallback):
    """Separate callback for validation progress"""
    
    def __init__(self, disable_progress: bool = False):
        self.pbar = None
        self.val_loss = 0
        self.disable_progress = disable_progress
    
    def on_validation_start(self, trainer, state: TrainingState) -> None:
        if not self.disable_progress:
            val_batches = len(trainer.datasets["val"]) if "val" in trainer.datasets else 0
            self.pbar = tqdm(
                total=val_batches,
                unit="batches",
                ascii=True,
                dynamic_ncols=True,
                disable=not trainer.accelerator.is_local_main_process,
            )
            self.pbar.set_description("Validation...")
            self.val_loss = 0
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if (self.pbar is not None and state.loss is not None and 
            state.stage == Stage.val):
            self.val_loss += state.loss.item()
            self.pbar.set_postfix(loss=self.val_loss / (state.batch_idx + 1))
            self.pbar.update(1)
    
    def on_validation_end(self, trainer, state: TrainingState) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class MetricsCallback(TrainingCallback):
    """Built-in callback for metrics computation"""
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if state.stage == Stage.train:
            for m in trainer.metrics:
                if ((state.epoch) % m.eval_period == 0 and not m.eval_only):
                    m(state.outputs, state.batch, Stage.train, trainer.device)
        elif state.stage == Stage.val:
            for m in trainer.metrics:
                if (state.epoch) % m.eval_period == 0:
                    m(state.outputs, state.batch, Stage.val, trainer.device)


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback based on validation loss"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        if state.metrics is None or self.monitor not in state.metrics:
            return
        
        current = state.metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current
        elif current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = state.epoch
                self.should_stop = True
                logger.info(f"Early stopping at epoch {state.epoch}")
    
    def should_stop_training(self) -> bool:
        return self.should_stop


class LearningRateSchedulerCallback(TrainingCallback):
    """Callback to handle learning rate scheduling"""
    
    def __init__(self, step_on: str = 'batch'):
        """
        Arguments:
        ----------
        step_on : str
            When to step the scheduler ('batch' or 'epoch')
        """
        assert step_on in ['batch', 'epoch'], "step_on must be 'batch' or 'epoch'"
        self.step_on = step_on
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if (self.step_on == 'batch' and hasattr(trainer, 'lr_sched') and 
            state.stage == Stage.train):
            trainer.lr_sched.step()
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        if self.step_on == 'epoch' and hasattr(trainer, 'lr_sched'):
            trainer.lr_sched.step()


class ModelSavingCallback(TrainingCallback):
    """Callback for saving model at specific intervals"""
    
    def __init__(self, save_every: int = 10, save_dir: str = "models"):
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        if state.epoch % self.save_every == 0 and trainer.accelerator.is_local_main_process:
            save_path = self.save_dir / f"model_epoch_{state.epoch}.pt"
            torch.save({
                'epoch': state.epoch,
                'model_state_dict': {k: v.state_dict() for k, v in trainer.modules.items()},
                'optimizer_state_dict': trainer.opt.state_dict() if hasattr(trainer, 'opt') else None,
                'metrics': state.metrics
            }, save_path)
            logger.info(f"Model saved at epoch {state.epoch} to {save_path}")


class GradientClippingCallback(TrainingCallback):
    """Callback for gradient clipping"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_backward_end(self, trainer, state: TrainingState) -> None:
        if state.stage == Stage.train:
            # Unscale gradients if using mixed precision
            trainer.accelerator.unscale_gradients(trainer.opt)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                trainer.modules.parameters(), 
                self.max_norm, 
                norm_type=self.norm_type
            )


class WarmupCallback(TrainingCallback):
    """Learning rate warmup callback"""
    
    def __init__(self, warmup_epochs: int = 5, base_lr: float = None):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.initial_lr = None
    
    def on_train_start(self, trainer, state: TrainingState) -> None:
        if self.base_lr is None:
            self.base_lr = trainer.hparams.lr
        self.initial_lr = self.base_lr / self.warmup_epochs
        
        # Set initial learning rate
        for param_group in trainer.opt.param_groups:
            param_group['lr'] = self.initial_lr
    
    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        if state.epoch <= self.warmup_epochs:
            lr = self.base_lr * (state.epoch / self.warmup_epochs)
            for param_group in trainer.opt.param_groups:
                param_group['lr'] = lr
            
            if trainer.accelerator.is_local_main_process:
                logger.info(f"Warmup LR: {lr:.6f}")


class LoggingCallback(TrainingCallback):
    """Enhanced logging callback for better training visibility"""
    
    def __init__(self, log_every: int = 100, log_metrics: bool = True):
        self.log_every = log_every
        self.log_metrics = log_metrics
    
    def on_epoch_end(self, trainer, state: TrainingState) -> None:
        if trainer.accelerator.is_local_main_process and state.metrics:
            # Log epoch metrics
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in state.metrics.items()])
            logger.info(f"Epoch {state.epoch}/{state.total_epochs} - {metrics_str}")
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if (trainer.accelerator.is_local_main_process and 
            state.batch_idx % self.log_every == 0 and 
            state.stage == Stage.train):
            
            lr = trainer.opt.param_groups[0]['lr'] if hasattr(trainer, 'opt') else 0.0
            logger.info(f"Epoch {state.epoch} - Batch {state.batch_idx}/{state.total_batches} - "
                       f"Loss: {state.loss.item():.4f} - LR: {lr:.6f}")


class MemoryTrackingCallback(TrainingCallback):
    """Callback to track GPU memory usage"""
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
    
    def on_batch_end(self, trainer, state: TrainingState) -> None:
        if (torch.cuda.is_available() and 
            trainer.accelerator.is_local_main_process and
            state.batch_idx % self.log_every == 0):
            
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                       f"Reserved: {memory_reserved:.2f}GB")