from __future__ import annotations

from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from ..utils.config import DatasetConfig
from ..utils.seed import worker_init_fn
from .datasets import build_datasets
from .transforms import load_augmentation


def create_dataloaders(
    cfg: DatasetConfig,
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, float]]:
    aug_params = load_augmentation(cfg.aug)
    train_ds, val_ds, class_names, aug = build_datasets(cfg, aug_params)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, class_names, {
        "mixup": aug.mixup_alpha,
        "cutmix": aug.cutmix_alpha,
    }
