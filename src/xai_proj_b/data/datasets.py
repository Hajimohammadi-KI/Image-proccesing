from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision import datasets

from ..utils.config import DatasetConfig
from .transforms import AugmentationParams, build_eval_transforms, build_train_transforms

OWN_DATASET_CLASSES = [
    "coffee_mug",
    "wooden_spoon",
    "notebook",
    "teapot",
    "soup_bowl",
    "remote_control",
    "computer_keyboard",
    "mouse",
    "toilet_tissue",
    "binder",
]

OWN_PATTERN = re.compile(
    r"^(?P<student>[a-zA-Z0-9]+)_(?P<phone>[a-zA-Z0-9]+)_(?P<classname>[a-zA-Z]+(?:[-_][a-zA-Z]+)*)_(?P<imageid>[0-9]+)\.jpe?g$",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    return name.replace("-", "_").lower()


def build_datasets(
    cfg: DatasetConfig,
    aug: AugmentationParams,
) -> Tuple[Dataset, Dataset, List[str], AugmentationParams]:
    dataset_name = cfg.name.lower()
    img_size = cfg.img_size or (32 if dataset_name == "cifar10" else 224)
    train_tf = build_train_transforms(dataset_name, img_size, aug)
    eval_tf = build_eval_transforms(dataset_name, img_size)

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(
            root=cfg.root,
            train=True,
            download=cfg.download,
            transform=train_tf,
        )
        val_ds = datasets.CIFAR10(
            root=cfg.root,
            train=False,
            download=cfg.download,
            transform=eval_tf,
        )
        return train_ds, val_ds, train_ds.classes, aug

    root = Path(cfg.root)
    train_dir = root / cfg.train_subdir
    val_dir = root / cfg.val_subdir
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train dir '{train_dir}' and val dir '{val_dir}'. Run scripts/prepare_imagenet_subset.py first."
        )

    train_ds = datasets.ImageFolder(train_dir.as_posix(), transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir.as_posix(), transform=eval_tf)

    if dataset_name == "own_dataset":
        _validate_own_dataset(train_dir)
        _validate_own_dataset(val_dir)

    return train_ds, val_ds, train_ds.classes, aug


def _validate_own_dataset(directory: Path) -> None:
    for class_dir in directory.iterdir():
        if not class_dir.is_dir():
            continue
        normalized = normalize_name(class_dir.name)
        if normalized not in OWN_DATASET_CLASSES:
            raise ValueError(f"Unexpected class '{class_dir.name}' in {directory}")
        for img_path in class_dir.glob("*.jp*g"):
            if not OWN_PATTERN.match(img_path.name):
                raise ValueError(f"Invalid filename '{img_path.name}' in {img_path.parent}")
