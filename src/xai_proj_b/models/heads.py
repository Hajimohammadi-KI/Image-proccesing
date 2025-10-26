from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

