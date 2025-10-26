import argparse
from pathlib import Path

from xai_proj_b.data.datasets import (
    OWN_DATASET_CLASSES,
    _validate_own_dataset,
    normalize_class_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate own_dataset structure and quotas.")
    parser.add_argument("--data-root", default="data/own_dataset", help="Root directory containing class folders or train/val subfolders.")
    parser.add_argument("--per-class-min", type=int, default=30)
    return parser.parse_args()


def count_images(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.jp*g"))


def main():
    args = parse_args()
    root = Path(args.data_root)
    errors = []
    targets = [root]
    if (root / "train").exists():
        targets = [root / "train", root / "val"]
    for split_dir in targets:
        if split_dir.exists():
            _validate_own_dataset(split_dir)
            normalized_dirs = {
                normalize_class_name(d.name): d
                for d in split_dir.iterdir()
                if d.is_dir()
            }
            for cls in OWN_DATASET_CLASSES:
                class_dir = normalized_dirs.get(cls)
                if class_dir is None:
                    errors.append(f"Missing class {cls} in {split_dir}")
                    continue
                if count_images(class_dir) < args.per_class_min:
                    errors.append(
                        f"Class {cls} in {split_dir} has fewer than {args.per_class_min} images"
                    )
    if errors:
        raise SystemExit("\n".join(errors))
    print("own_dataset validation passed OK")


if __name__ == "__main__":
    main()
