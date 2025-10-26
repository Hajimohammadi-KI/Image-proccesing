from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    best_score: float | None = None
    counter: int = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

