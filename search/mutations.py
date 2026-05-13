from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseMutation:
    latent_indices: list[int]
    coefficients: torch.Tensor
    delta: torch.Tensor
    parent_score: float | None = None
    child_score: float | None = None
