from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PerplexityRegressionWeights:
    weights: torch.Tensor
    bias: float
    target_mean: float
    target_std: float
    target_key: str
    normalized: bool = True
    layer: int | None = None
    token_pos: int | None = None

    def predict(self, h_sparse: torch.Tensor) -> torch.Tensor:
        features = h_sparse.to(dtype=self.weights.dtype, device=self.weights.device)
        pred = (features * self.weights).sum(dim=-1) + self.bias
        if self.normalized and self.target_std != 0.0:
            pred = pred * self.target_std + self.target_mean
        return pred


def save_perplexity_regression_weights(
    weights: PerplexityRegressionWeights,
    path: Path,
) -> None:
    payload = {
        "weights": weights.weights.detach().cpu(),
        "bias": float(weights.bias),
        "target_mean": float(weights.target_mean),
        "target_std": float(weights.target_std),
        "target_key": weights.target_key,
        "normalized": weights.normalized,
        "layer": weights.layer,
        "token_pos": weights.token_pos,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_perplexity_regression_weights(
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> PerplexityRegressionWeights:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weights = payload["weights"]
    if dtype is not None:
        weights = weights.to(dtype=dtype)
    weights = weights.to(device=device)
    return PerplexityRegressionWeights(
        weights=weights,
        bias=float(payload.get("bias", 0.0)),
        target_mean=float(payload.get("target_mean", 0.0)),
        target_std=float(payload.get("target_std", 1.0)),
        target_key=str(payload.get("target_key", "perplexity")),
        normalized=bool(payload.get("normalized", True)),
        layer=payload.get("layer"),
        token_pos=payload.get("token_pos"),
    )
