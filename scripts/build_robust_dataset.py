import argparse
import json

from xai_proj_b.eval.robustness_dataset import generate_robust_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corruption-heavy robustness dataset.")
    parser.add_argument("--base-dataset", default="cifar10")
    parser.add_argument("--base-root", default="data/cifar10")
    parser.add_argument("--output-dir", default="data/robustness_dataset")
    parser.add_argument("--samples-per-class", type=int, default=400)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    path = generate_robust_dataset(
        base_dataset=args.base_dataset,
        base_root=args.base_root,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(path)}))


if __name__ == "__main__":
    main()
