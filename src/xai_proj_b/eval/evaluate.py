from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from ..data.loaders import create_dataloaders
from ..models.factory import create_model
from ..train.metrics import MetricTracker, plot_confusion
from ..utils.config import ExperimentConfig
from ..utils.seed import set_seed


def evaluate_checkpoint(
    cfg: ExperimentConfig,
    checkpoint_path: str,
    dataset_name: Optional[str] = None,
    data_root: Optional[str] = None,
) -> dict:
    if dataset_name:
        cfg.dataset.name = dataset_name
    if data_root:
        cfg.dataset.root = data_root
    set_seed(cfg.seeds[0] if cfg.seeds else 0, cfg.train.deterministic)
    _, val_loader, class_names, _ = create_dataloaders(cfg.dataset)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    tracker = MetricTracker(num_classes, device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            tracker.update(outputs, targets, loss)

    results = tracker.compute()
    output = {
        "val_loss": results.loss,
        **{f"val_{k}": v for k, v in results.metrics.items()},
    }
    plots_dir = Path(cfg.expanded_output_dir()) / "eval"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(results.confusion, class_names, plots_dir / "confusion_eval.png")
    (plots_dir / "per_class.json").write_text(
        json.dumps(
            {
                "classes": class_names,
                "precision": results.per_class_precision,
                "recall": results.per_class_recall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = plots_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

