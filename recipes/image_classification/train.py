"""
This code runs the image classification training loop. It tries to support as much
as timm's functionalities as possible.

For compatibility the prefetcher, re_split and JSDLoss are disabled.

To run the training script, use this command:
    python train.py cfg/phinet.py

You can change the configuration or override the parameters as you see fit.

Authors:
    - Francesco Paissan, 2023
"""

import torch
import torch.nn as nn
from prepare_data import ImageDataModule, setup_mixup
from timm.loss import (
    BinaryCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

import micromind as mm
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration
import sys


class ImageClassification(mm.MicroMind):
    """
    Implements an image classification class.
    Provides support for timm augmentation and loss functions.
    """

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        # Build model
        if hparams.model == "phinet":
            self.modules_dict["classifier"] = PhiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_layers=hparams.num_layers,
                beta=hparams.beta,
                t_zero=hparams.t_zero,
                compatibility=False,
                divisor=hparams.divisor,
                downsampling_layers=hparams.downsampling_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
            )
        elif hparams.model == "xinet":
            self.modules_dict["classifier"] = XiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                gamma=hparams.gamma,
                num_layers=hparams.num_layers,
                return_layers=hparams.return_layers,
                include_top=True,
                num_classes=hparams.num_classes,
            )

        self.mixup_fn, _ = setup_mixup(hparams)
        self.criterion = self.setup_criterion()

        self.compute_params() # Number of parameters
        self.compute_macs(hparams.input_shape) # Number of MACs

    def setup_criterion(self):
        """Setup of the loss function based on augmentation strategy."""
        # setup loss function
        if (
            self.hparams.mixup > 0
            or self.hparams.cutmix > 0.0
            or self.hparams.cutmix_minmax is not None
        ):
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.hparams.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.hparams.smoothing:
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.hparams.smoothing,
                    target_threshold=self.hparams.bce_target_thresh,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.hparams.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn

    def forward(self, batch):
        """
        Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        img, target = batch
        
        # Apply mixup if available and in training mode
        if self.mixup_fn is not None and self.training:
            img, target = self.mixup_fn(img, target)

        return (self.modules_dict["classifier"](img), target)

    def compute_loss(self, pred, batch):
        """
        Computes the loss function.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        lr = getattr(self.hparams, 'lr', 3e-4)
        weight_decay = getattr(self.hparams, 'weight_decay', 0.0005)
        
        opt = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

# ==============================================================================
# TRAINING SCRIPT
# ==============================================================================

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    # Optional: Set resource targets
    # resources = {
    #     "WM": 512 * 1024 * 1024,  # in bytes
    #     "FLASH": hparams.target_FLASH,
    #     "MACCs": hparams.target_MACCs,
    # }

    # Create the data module
    datamodule = ImageDataModule(hparams)

    # Initialize model
    mind = ImageClassification(hparams=hparams)

    # Add metrics (automatically tracked on train/val/test)
    mind.add_metric(
        'acc',
        Accuracy(task='multiclass', num_classes=hparams.num_classes),
        stage='all'
    )

    trainer = mind.fit(
        datamodule,
        auto_resume=True,  # Automatically resume from last checkpoint
    )

    # Test using the best checkpoint
    results = mind.test(
        datamodule,
        ckpt_path="best"  # Can also use "last" or path to specific checkpoint
    )

    print(f"\n{'='*60}")
    print(f"Training complete!")
    if mind._checkpoint_callback:
        print(f"Best checkpoint: {mind._checkpoint_callback.best_model_path}")
    print(f"Test results: {results}")
    print(f"{'='*60}\n")