from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import datasets, transforms
from tqdm import tqdm


def _load_dataset(name: str, root: str):
    if name.lower() == "cifar10":
        return datasets.CIFAR10(root=root, train=False, download=True)
    raise ValueError(f"Unsupported base dataset '{name}'.")


def _jpeg(img: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=random.randint(10, 40))
    buffer.seek(0)
    degraded = Image.open(buffer).convert("RGB")
    buffer.close()
    return degraded


def _gaussian_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 25, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _occlusion(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w, _ = arr.shape
    size = random.randint(h // 6, h // 3)
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    arr[y : y + size, x : x + size] = np.random.randint(0, 255, (size, size, 3))
    return Image.fromarray(arr)


def _solarize(img: Image.Image) -> Image.Image:
    return ImageOps.solarize(img, threshold=random.randint(64, 192))


def _posterize(img: Image.Image) -> Image.Image:
    bits = random.choice([1, 2, 3, 4])
    return ImageOps.posterize(img, bits=bits)


def _blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))


CORRUPTIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg": _jpeg,
    "gaussian_noise": _gaussian_noise,
    "occlusion": _occlusion,
    "solarize": _solarize,
    "posterize": _posterize,
    "blur": _blur,
}


def generate_robust_dataset(
    base_dataset: str,
    base_root: str,
    output_dir: str,
    samples_per_class: int = 400,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    dataset = _load_dataset(base_dataset, base_root)
    class_names = dataset.classes
    transform = transforms.Resize(160)
    output_root = Path(output_dir)
    for split in ("train", "val"):
        (output_root / split).mkdir(parents=True, exist_ok=True)
    metadata: List[dict] = []
    per_class_counts = {cls: 0 for cls in class_names}

    for idx in tqdm(range(len(dataset)), desc="Corrupting dataset"):
        image, label = dataset[idx]
        cls_name = class_names[label]
        if per_class_counts[cls_name] >= samples_per_class:
            continue
        corruption_name, corruption_fn = random.choice(list(CORRUPTIONS.items()))
        corrupted = corruption_fn(image)
        corrupted = transform(corrupted)
        split = "val" if random.random() < val_ratio else "train"
        cls_dir = output_root / split / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{cls_name}_{per_class_counts[cls_name]:05d}_{corruption_name}.png"
        full_path = cls_dir / filename
        corrupted.save(full_path)
        metadata.append(
            {
                "path": str(full_path.relative_to(output_root)),
                "class": cls_name,
                "label": int(label),
                "corruption": corruption_name,
            }
        )
        per_class_counts[cls_name] += 1
        if all(count >= samples_per_class for count in per_class_counts.values()):
            break

    import pandas as pd

    pd.DataFrame(metadata).to_csv(output_root / "metadata.csv", index=False)
    return output_root

