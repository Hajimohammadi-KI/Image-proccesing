from __future__ import annotations

import timm
import torch.nn as nn

from ..utils.config import ModelConfig


def create_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        num_classes=num_classes,
        drop_rate=cfg.dropout,
    )
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    return model

