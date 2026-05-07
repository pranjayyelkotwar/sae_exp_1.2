from __future__ import annotations

from pathlib import Path

import torch


class GroundingScoreCalculator:
    def __init__(self, c_vector: torch.Tensor, eps: float = 1e-6) -> None:
        self.c_vector = c_vector
        self.eps = eps

    @classmethod
    def from_avg_latents(
        cls,
        avg_latents_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        eps: float = 1e-6,
    ) -> "GroundingScoreCalculator":
        arc_path = avg_latents_dir / "avg_latents_arc_easy.pt"
        hle_path = avg_latents_dir / "avg_latents_hle.pt"

        if not arc_path.exists() or not hle_path.exists():
            raise FileNotFoundError(
                "Missing avg latents; expected ARC-Easy and HLE tensors in avg_latents_dir"
            )

        arc = torch.load(arc_path, map_location="cpu", weights_only=True).to(
            device=device, dtype=dtype
        )
        hle = torch.load(hle_path, map_location="cpu", weights_only=True).to(
            device=device, dtype=dtype
        )

        if arc.shape != hle.shape:
            raise ValueError(
                f"Shape mismatch: ARC-Easy {tuple(arc.shape)} vs HLE {tuple(hle.shape)}"
            )

        c_vector = arc - hle
        return cls(c_vector=c_vector, eps=eps)

    def score(self, h_sparse: torch.Tensor) -> torch.Tensor:
        return (self.c_vector * h_sparse).sum(dim=-1) / (h_sparse.sum(dim=-1) + self.eps)
