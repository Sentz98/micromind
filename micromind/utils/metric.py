import torch
from typing import Callable
from torchmetrics import Metric as TorchMetric


class CustomMetric(TorchMetric):
    """
    Wrapper to convert a user-defined function into a TorchMetric.
    
    This allows users to define custom metrics while maintaining all the
    benefits of TorchMetrics (distributed sync, state management, etc.).
    
    Arguments
    ---------
        compute_fn : Callable
            Function that takes (predictions, targets) and returns a metric value.
            Can return a scalar tensor, float, or multi-element tensor.
        reduction : str
            How to aggregate values across batches ('mean' or 'sum'). Default: 'mean'
        higher_is_better : bool
            Whether higher values are better for this metric. Default: True
    
    Example
    -------
    >>> def dice_score(preds, targets):
    ...     intersection = (preds * targets).sum()
    ...     union = preds.sum() + targets.sum()
    ...     return 2 * intersection / (union + 1e-8)
    >>> 
    >>> metric = CustomMetric(dice_score, reduction='mean', higher_is_better=True)
    >>> # Now use like any TorchMetric
    >>> metric.update(preds, targets)
    >>> score = metric.compute()
    
    Note
    ----
    Your compute_fn should:
    - Accept (predictions, targets) as arguments
    - Return a scalar or tensor value
    - Be differentiable if you want to use it in loss
    - Handle batch dimensions appropriately
    """
    
    def __init__(
        self, 
        compute_fn: Callable,
        reduction: str = "mean",
        higher_is_better: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compute_fn = compute_fn
        self.reduction = reduction
        self.higher_is_better = higher_is_better
        
        # State for accumulation across batches
        # Using dist_reduce_fx="sum" ensures proper synchronization across GPUs
        self.add_state("values", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metric state with batch predictions and targets.
        
        This is called automatically by Lightning for each batch.
        """
        # Compute metric for this batch
        value = self.compute_fn(preds, target)
        
        # Ensure it's a tensor
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=preds.device, dtype=torch.float32)
        
        # Handle multi-element results
        if value.numel() > 1:
            if self.reduction == "mean":
                value = value.mean()
            elif self.reduction == "sum":
                value = value.sum()
            else:
                raise ValueError(f"Unsupported reduction: {self.reduction}")
        
        # Ensure scalar
        value = value.view(-1)
        
        # Accumulate
        self.values.append(value)
    
    def compute(self) -> torch.Tensor:
        """
        Compute final metric value from accumulated state.
        
        This is called automatically by Lightning at the end of epoch.
        """
        if len(self.values) == 0:
            return torch.tensor(0.0)
        
        # Concatenate all batch values
        values = torch.cat(self.values)
        
        # Apply reduction
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
