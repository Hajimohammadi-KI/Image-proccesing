from __future__ import annotations

from .evaluate import evaluate_checkpoint


def run_robustness(cfg, checkpoint_path: str, data_root: str):
    cfg.dataset.name = "own_dataset"
    cfg.dataset.root = data_root
    return evaluate_checkpoint(cfg, checkpoint_path, dataset_name="own_dataset", data_root=data_root)

