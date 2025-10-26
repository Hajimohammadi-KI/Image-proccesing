from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import yaml
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy, InterpolationMode


DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "imagenet_subset": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "own_dataset": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


@dataclass
class AugmentationParams:
    name: str = "baseline"
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float | None = None
    auto_augment: str | None = None
    random_erasing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0


def load_augmentation(path: str | None) -> AugmentationParams:
    if not path:
        return AugmentationParams()
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return AugmentationParams(**payload)


def _stats(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return DATASET_STATS.get(dataset_name, DATASET_STATS["imagenet_subset"])


def build_train_transforms(dataset_name: str, img_size: int, aug: AugmentationParams) -> transforms.Compose:
    mean, std = _stats(dataset_name)
    ops: list[transforms.Transform] = []

    if aug.random_crop:
        if dataset_name == "cifar10" and img_size <= 64:
            ops.append(transforms.RandomCrop(32, padding=4, padding_mode="reflect"))
            if img_size != 32:
                ops.append(transforms.Resize(img_size, InterpolationMode.BICUBIC))
        else:
            ops.append(
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.6, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
    else:
        ops.append(transforms.Resize(img_size, InterpolationMode.BICUBIC))

    if aug.random_flip:
        ops.append(transforms.RandomHorizontalFlip())

    if aug.color_jitter:
        ops.append(
            transforms.ColorJitter(
                aug.color_jitter,
                aug.color_jitter,
                aug.color_jitter,
                min(0.1, aug.color_jitter / 2),
            )
        )

    if aug.auto_augment:
        policy = AutoAugmentPolicy.CIFAR10 if dataset_name == "cifar10" else AutoAugmentPolicy.IMAGENET
        ops.append(transforms.AutoAugment(policy=policy))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if aug.random_erasing > 0:
        ops.append(transforms.RandomErasing(p=aug.random_erasing, value="random"))

    return transforms.Compose(ops)


def build_eval_transforms(dataset_name: str, img_size: int) -> transforms.Compose:
    mean, std = _stats(dataset_name)
    ops: list[transforms.Transform] = []
    if dataset_name == "cifar10" and img_size <= 64:
        ops.append(transforms.Resize(img_size, InterpolationMode.BICUBIC))
    else:
        resize_size = int(img_size * 1.15)
        ops.extend(
            [
                transforms.Resize(resize_size, InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)

