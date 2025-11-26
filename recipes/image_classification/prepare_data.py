"""
This code prepares the DataLoader compatible with HF accelerate and exploiting
timm data augmentation.
For compatibility, the prefetcher, JSDLoss and re_split options where disabled.

Authors:
    - Francesco Paissan, 2023

"""
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from timm.data import (
    AugMixDataset,
    Mixup,
    create_dataset,
    create_transform,
)
from argparse import Namespace


def setup_mixup(args: Namespace):
    """Setup Mixup data augmentation."""
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None

    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        mixup_fn = Mixup(**mixup_args)

    return mixup_fn, collate_fn


class ImageDataModule(LightningDataModule):
    """
    LightningDataModule version timm dataloader pipeline.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.dataset_train = None
        self.dataset_eval = None
        self.mixup_fn = None
        self.collate_fn = None

        # force-disable options
        self.args.prefetcher = False
        self.args.distributed = False

        # aug splits
        if args.aug_splits > 1:
            self.num_aug_splits = args.aug_splits
        else:
            self.num_aug_splits = 0

    def setup(self, stage=None):
        """Create datasets and transforms."""
        args = self.args

        # datasets
        self.dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
        )
        self.dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
        )

        # mixup
        self.mixup_fn, self.collate_fn = setup_mixup(args)

        # wrap dataset in AugMix helper
        if self.num_aug_splits > 1:
            self.dataset_train = AugMixDataset(self.dataset_train, num_splits=self.num_aug_splits)

        # transforms
        train_interp = args.train_interpolation or args.interpolation
        re_splits = 0

        self.dataset_train.transform = create_transform(
            input_size=args.input_shape,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=train_interp,
            mean=args.mean,
            std=args.std,
            tf_preprocessing=False,
            re_num_splits=re_splits,
            separate=self.num_aug_splits > 0,
        )

        self.dataset_eval.transform = create_transform(
            input_size=args.input_shape,
            is_training=False,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=train_interp,
            mean=args.mean,
            std=args.std,
            tf_preprocessing=False,
            re_num_splits=re_splits,
            separate=self.num_aug_splits > 0,
        )

        # default collate
        if self.collate_fn is None:
            self.collate_fn = torch.utils.data.dataloader.default_collate

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_memory,
            drop_last=True,
            persistent_workers=self.args.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_eval,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_memory,
            drop_last=False,
            persistent_workers=self.args.persistent_workers,
        )

