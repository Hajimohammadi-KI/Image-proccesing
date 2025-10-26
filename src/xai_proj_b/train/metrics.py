from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)


@dataclass
class MetricResults:
    loss: float
    metrics: Dict[str, float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    confusion: torch.Tensor


class MetricTracker:
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.acc1 = MulticlassAccuracy(num_classes=num_classes).to(device)
        self.acc5 = (
            MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
            if num_classes >= 5
            else None
        )
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        self.per_class_precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        self.per_class_recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        self.confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        self.running_loss = 0.0
        self.total = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor):
        batch_size = targets.size(0)
        self.running_loss += loss.item() * batch_size
        self.total += batch_size
        self.acc1.update(outputs, targets)
        if self.acc5:
            self.acc5.update(outputs, targets)
        self.precision.update(outputs, targets)
        self.recall.update(outputs, targets)
        self.per_class_precision.update(outputs, targets)
        self.per_class_recall.update(outputs, targets)
        self.confusion.update(outputs, targets)

    def compute(self) -> MetricResults:
        metrics = {
            "acc1": self.acc1.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
        }
        if self.acc5:
            metrics["acc5"] = self.acc5.compute().item()
        per_precision = self.per_class_precision.compute().cpu().tolist()
        per_recall = self.per_class_recall.compute().cpu().tolist()
        return MetricResults(
            loss=self.running_loss / max(1, self.total),
            metrics=metrics,
            per_class_precision=per_precision,
            per_class_recall=per_recall,
            confusion=self.confusion.compute().cpu(),
        )


def plot_confusion(confusion: torch.Tensor, class_names: List[str], path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion.numpy(), interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

