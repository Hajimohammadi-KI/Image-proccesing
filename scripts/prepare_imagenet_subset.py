import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ImageNet subset train/val folders.")
    parser.add_argument("--input-dir", required=True, help="Source directory with class subfolders.")
    parser.add_argument("--output-dir", default="data/imagenet_subset", help="Destination root.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of move.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    train_root = output_root / "train"
    val_root = output_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        images = list(class_dir.glob("*.jp*g"))
        if not images:
            continue
        random.shuffle(images)
        split_idx = int(len(images) * (1 - args.val_ratio))
        train_split = images[:split_idx]
        val_split = images[split_idx:]
        _transfer(train_split, train_root / class_dir.name, args.copy)
        _transfer(val_split, val_root / class_dir.name, args.copy)


def _transfer(files, dest_dir: Path, copy: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = dest_dir / file.name
        if copy:
            shutil.copy2(file, target)
        else:
            shutil.move(file, target)


if __name__ == "__main__":
    main()
