from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.loaders import create_dataloaders
from ..models.factory import create_model
from ..utils import checkpoint
from ..utils.config import ExperimentConfig, save_config
from ..utils.logging import init_wandb, setup_logging
from ..utils.system_stats import get_system_stats
from ..utils.seed import set_seed
from .callbacks import EarlyStopping
from .metrics import MetricTracker, plot_confusion


_PUBLIC_PROGRESS_OVERRIDE = os.getenv("XAI_PROGRESS_PUBLIC_PATH")
_PUBLIC_PROGRESS_PATH = None
if _PUBLIC_PROGRESS_OVERRIDE:
    _PUBLIC_PROGRESS_PATH = Path(_PUBLIC_PROGRESS_OVERRIDE)
else:
    for candidate in Path(__file__).resolve().parents:
        potential = candidate / "web" / "progress-dashboard" / "public" / "progress.json"
        if potential.parent.exists():
            _PUBLIC_PROGRESS_PATH = potential
            break


def run_training(cfg: ExperimentConfig) -> Dict[str, float]:
    output_root = cfg.expanded_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")
    seed_results = []
    summary_entries = []

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, torch.cuda.current_device())
        except RuntimeError:
            pass

    progress_path = run_dir / "progress.json"
    run_start = time.time()
    _write_progress(
        progress_path,
        {
            "status": "initializing",
            "current_epoch": 0,
            "total_epochs": cfg.sched.epochs,
            "train_loss": None,
            "val_loss": None,
            "val_acc1": None,
            "elapsed_sec": 0.0,
        },
    )

    for seed in cfg.seeds:
        last_val_loss: Optional[float] = None
        last_val_acc1: Optional[float] = None
        set_seed(seed, cfg.train.deterministic)
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(run_dir, cfg.experiment, seed)
        train_loader, val_loader, class_names, aug_mix = create_dataloaders(cfg.dataset)
        num_classes = len(class_names)
        model = create_model(cfg.model, num_classes=num_classes).to(_device())

        optimizer = _build_optimizer(model, cfg)
        scheduler = _build_scheduler(optimizer, cfg)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing).to(_device())
        scaler = GradScaler(enabled=cfg.train.amp and torch.cuda.is_available())
        callbacks = EarlyStopping(cfg.train.early_stop_patience) if cfg.train.early_stop_patience else None
        wandb_run = init_wandb(cfg.logging, f"{cfg.experiment}-{timestamp}", seed, {"config": cfg.experiment})

        history = []
        best_metric = -float("inf")
        best_state = None

        for epoch in range(1, cfg.sched.epochs + 1):
            def update_progress(batch_idx: int, total_batches: int, avg_loss: float, avg_acc: float) -> None:
                fractional_epoch = (epoch - 1) + (batch_idx / max(1, total_batches))
                _write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "current_epoch": fractional_epoch,
                        "total_epochs": cfg.sched.epochs,
                        "train_loss": avg_loss,
                        "val_loss": last_val_loss,
                        "val_acc1": last_val_acc1,
                        "elapsed_sec": time.time() - run_start,
                    },
                )

            train_stats = _train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                cfg,
                aug_mix,
                progress_hook=update_progress,
            )
            val_stats = _validate(model, val_loader, criterion, num_classes)
            scheduler_step(scheduler, val_stats.metrics.get("acc1", 0.0))

            last_val_loss = val_stats.loss
            last_val_acc1 = val_stats.metrics.get("acc1")

            metrics = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats.loss,
                **{f"val_{k}": v for k, v in val_stats.metrics.items()},
            }
            history.append(metrics)
            logger.info("Epoch %s | %s", epoch, metrics)
            if wandb_run:
                wandb_run.log(metrics)

            _write_progress(
                progress_path,
                {
                    "status": "running",
                    "current_epoch": epoch,
                    "total_epochs": cfg.sched.epochs,
                    "train_loss": metrics["train_loss"],
                    "val_loss": metrics["val_loss"],
                    "val_acc1": metrics.get("val_acc1"),
                    "elapsed_sec": time.time() - run_start,
                },
            )

            if val_stats.metrics.get("acc1", 0.0) > best_metric:
                best_metric = val_stats.metrics["acc1"]
                best_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "class_names": class_names,
                }
                plot_confusion(val_stats.confusion, class_names, seed_dir / "confusion.png")
                (seed_dir / "per_class.json").write_text(
                    json.dumps(
                        {
                            "classes": class_names,
                            "precision": val_stats.per_class_precision,
                            "recall": val_stats.per_class_recall,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                checkpoint.save_checkpoint(best_state, seed_dir / "best.pt")

            if callbacks and callbacks.step(val_stats.metrics.get("acc1", 0.0)):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        checkpoint.save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": history[-1]["epoch"],
                "metrics": history[-1],
            },
            seed_dir / "last.pt",
        )
        _write_history(history, seed_dir / "metrics.csv")
        per_seed_summary = {
            "seed": seed,
            "best_acc1": best_metric,
            "best_epoch": best_state["metrics"]["epoch"] if best_state else history[-1]["epoch"],
            "val_precision": best_state["metrics"].get("val_precision", 0.0) if best_state else history[-1].get("val_precision", 0.0),
            "val_recall": best_state["metrics"].get("val_recall", 0.0) if best_state else history[-1].get("val_recall", 0.0),
        }
        seed_results.append(per_seed_summary)
        summary_entries.append(best_state["metrics"] if best_state else history[-1])
        if wandb_run:
            wandb_run.finish()

    summary = _aggregate_results(seed_results)
    summary_payload = {"seeds": seed_results, "aggregate": summary}
    summary_path = run_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary_md(run_dir / "SUMMARY.md", cfg, summary_payload)
    latest_metrics = summary_entries[-1] if summary_entries else {}
    _write_progress(
        progress_path,
        {
            "status": "completed",
            "current_epoch": cfg.sched.epochs,
            "total_epochs": cfg.sched.epochs,
            "train_loss": latest_metrics.get("train_loss"),
            "val_loss": latest_metrics.get("val_loss"),
            "val_acc1": latest_metrics.get("val_acc1"),
            "elapsed_sec": time.time() - run_start,
        },
    )
    return summary


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optim.name.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum or 0.9,
            weight_decay=cfg.optim.weight_decay,
            nesterov=True,
        )
    if cfg.optim.name.lower() == "adam":
        betas = cast(Tuple[float, float], tuple(cfg.optim.betas or (0.9, 0.999)))
        return torch.optim.Adam(
            params,
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=betas,
        )
    betas = cast(Tuple[float, float], tuple(cfg.optim.betas or (0.9, 0.999)))
    return torch.optim.AdamW(
        params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=betas,
    )


def _build_scheduler(optimizer, cfg: ExperimentConfig):
    name = cfg.sched.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.sched.epochs,
            eta_min=cfg.sched.min_lr,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.sched.step_size,
            gamma=cfg.sched.gamma,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.sched.gamma,
            patience=cfg.sched.step_size,
        )
    return None


def scheduler_step(scheduler, metric_value: float):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric_value)
    else:
        scheduler.step()


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    cfg: ExperimentConfig,
    aug_mix: Dict[str, float],
    progress_hook: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    device = _device()
    mixup_alpha = aug_mix.get("mixup", 0.0)
    cutmix_alpha = aug_mix.get("cutmix", 0.0)

    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.train.amp and torch.cuda.is_available()):
            if mixup_alpha > 0 or cutmix_alpha > 0:
                images, targets_a, targets_b, lam = _apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        if cfg.train.grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        if progress_hook:
            avg_loss = total_loss / max(1, total_samples)
            avg_acc = total_correct / max(1, total_samples)
            progress_hook(batch_idx, total_batches, avg_loss, avg_acc)

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples),
    }


def _validate(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, num_classes: int):
    model.eval()
    tracker = MetricTracker(num_classes, _device())
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(_device(), non_blocking=True)
            targets = targets.to(_device(), non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            tracker.update(outputs, targets, loss)
    return tracker.compute()


def _apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha):
    beta = torch.distributions.Beta(
        torch.tensor([mixup_alpha]), torch.tensor([mixup_alpha])
    ) if mixup_alpha > 0 else None
    cutmix_beta = torch.distributions.Beta(
        torch.tensor([cutmix_alpha]), torch.tensor([cutmix_alpha])
    ) if cutmix_alpha > 0 else None

    if mixup_alpha > 0 and cutmix_alpha > 0:
        use_mixup = torch.rand(1).item() > 0.5
    else:
        use_mixup = mixup_alpha > 0

    if use_mixup:
        lam = beta.sample().item() if beta else 1.0
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[index, :]
        return mixed, targets, targets[index], lam

    lam = cutmix_beta.sample().item() if cutmix_beta else 1.0
    index = torch.randperm(images.size(0), device=images.device)
    bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(2), images.size(3), lam)
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    return images, targets, targets[index], lam


def _rand_bbox(height, width, lam):
    cut_ratio = torch.sqrt(torch.tensor(1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = torch.randint(width, size=(1,)).item()
    cy = torch.randint(height, size=(1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


def _write_history(history: List[Dict[str, float]], path: Path):
    import pandas as pd

    df = pd.DataFrame(history)
    df.to_csv(path, index=False)


def _aggregate_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    import numpy as np

    metrics = defaultdict(list)
    for res in results:
        for key, value in res.items():
            if key == "seed":
                continue
            metrics[key].append(value)
    aggregate = {}
    for key, values in metrics.items():
        arr = np.array(values)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std())
    return aggregate


def _write_summary_md(path: Path, cfg: ExperimentConfig, payload: Dict[str, Any]) -> None:
    lines = [
        f"# Summary · {cfg.experiment}",
        "",
        f"- Dataset: `{cfg.dataset.name}`",
        f"- Model: `{cfg.model.name}`",
        f"- Seeds: {cfg.seeds}",
        "",
        "## Seed Metrics",
    ]
    for entry in payload["seeds"]:
        lines.append(f"- Seed {entry['seed']}: acc1={entry.get('best_acc1', 0):.4f}, precision={entry.get('val_precision', 0):.4f}, recall={entry.get('val_recall', 0):.4f}")
    lines.append("")
    lines.append("## Aggregate")
    for key, value in payload["aggregate"].items():
        lines.append(f"- {key}: {value:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_progress(path: Path, payload: Dict[str, Any]) -> None:
    stats = _collect_system_stats()
    merged = {**payload, **stats}
    safe_payload = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in merged.items()}
    path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    _mirror_progress(safe_payload)


def _mirror_progress(payload: Dict[str, Any]) -> None:
    if not _PUBLIC_PROGRESS_PATH:
        return
    try:
        _PUBLIC_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PUBLIC_PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        # If the mirror path cannot be written (e.g., permissions), just skip silently.
        pass


def _collect_system_stats() -> Dict[str, Any]:
    return get_system_stats()
