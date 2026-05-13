from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StoppingConfig:
    max_iters: int = 10
    min_improvement: float = 1e-4
    patience: int = 3
    target_grounding: float | None = None


@dataclass
class StoppingState:
    best_score: float
    steps_since_improve: int = 0
