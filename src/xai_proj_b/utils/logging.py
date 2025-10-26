from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, experiment: str, seed: int) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed}.log"
    logger = logging.getLogger(f"xai_proj_b.{experiment}.seed{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_wandb(logging_cfg, experiment: str, seed: int, config_dict: dict):
    if not logging_cfg.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Weights & Biases is enabled but not installed.") from exc
    run = wandb.init(
        project=logging_cfg.project or "xai-proj-b",
        entity=logging_cfg.entity,
        name=f"{experiment}-seed{seed}",
        config=config_dict,
    )
    return run

