from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .eval.evaluate import evaluate_checkpoint
from .eval.robustness import run_robustness
from .train.loop import run_training
from .utils.config import ExperimentConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xAI Proj B vision training CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a YAML config.")
    train_parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    train_parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds.")
    train_parser.add_argument("--aug", help="Override augmentation preset (weak/strong/custom path).")
    train_parser.add_argument("--dataset-root", help="Override dataset root path.")
    train_parser.add_argument("--output-dir", help="Override output directory.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a dataset.")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--dataset", default=None)
    eval_parser.add_argument("--data-root", default=None)

    rob_parser = subparsers.add_parser("robustness", help="Evaluate checkpoint on own_dataset.")
    rob_parser.add_argument("--config", required=True)
    rob_parser.add_argument("--checkpoint", required=True)
    rob_parser.add_argument("--data-root", required=True)

    sweep_parser = subparsers.add_parser("sweep", help="Run a sweep of experiments.")
    sweep_parser.add_argument("--config", required=True, help="Sweep YAML definition.")
    sweep_parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        cfg = _load_with_overrides(args.config, args.aug, args.dataset_root, args.output_dir, args.seeds)
        run_training(cfg)
    elif args.command == "evaluate":
        cfg = _load_with_overrides(args.config, None, args.data_root, None, None)
        results = evaluate_checkpoint(cfg, args.checkpoint, dataset_name=args.dataset, data_root=args.data_root)
        print(results)
    elif args.command == "robustness":
        cfg = _load_with_overrides(args.config, None, args.data_root, None, None)
        results = run_robustness(cfg, args.checkpoint, args.data_root)
        print(results)
    elif args.command == "sweep":
        _run_sweep(args.config, args.limit)


def _load_with_overrides(config_path, aug, dataset_root, output_dir, seeds):
    overrides = {}
    dataset_overrides = {}
    if aug:
        aug_path = aug
        preset = Path("configs/aug") / f"{aug}.yaml"
        if preset.exists():
            aug_path = str(preset)
        dataset_overrides["aug"] = aug_path
    if dataset_root:
        dataset_overrides["root"] = dataset_root
    if dataset_overrides:
        overrides["dataset"] = dataset_overrides
    if output_dir:
        overrides["output_dir"] = output_dir
    cfg = load_config(config_path, overrides)
    if seeds:
        cfg.seeds = seeds
    return cfg


def _run_sweep(sweep_path: str, limit: int | None):
    import yaml

    sweep = yaml.safe_load(Path(sweep_path).read_text(encoding="utf-8"))
    experiments = sweep.get("experiments", [])
    if limit:
        experiments = experiments[:limit]
    for exp in experiments:
        cfg_path = exp["config"]
        overrides = exp.get("overrides", {})
        cfg = load_config(cfg_path, overrides)
        run_training(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])

