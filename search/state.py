from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SearchState:
    prompt_tokens: list[int]
    hidden_state: torch.Tensor
    override_activations: torch.Tensor
    grounding_score: float
    step: int
    trajectory_id: str
    token_pos: int
    override_layer: int
    metadata: dict[str, Any] = field(default_factory=dict)
