from __future__ import annotations

from dataclasses import dataclass

import torch

from grounding_functions.curv import _build_decoder_direction
from search.mutations import SparseMutation


@dataclass
class SparseMutationSamplerConfig:
    num_candidates: int = 16
    min_active: int = 1
    max_active: int = 3
    beta: float = 0.01
    eps: float = 1e-8


class SparseMutationSampler:
    def __init__(
        self,
        config: SparseMutationSamplerConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config or SparseMutationSamplerConfig()
        self.generator = generator

    def sample(
        self,
        active_latents: list[int],
        decoder_weight: torch.Tensor,
        h_token: torch.Tensor,
    ) -> list[SparseMutation]:
        if not active_latents:
            raise ValueError("active_latents must be non-empty")

        device = decoder_weight.device
        dtype = decoder_weight.dtype
        d_model = decoder_weight.shape[0]
        max_active = min(self.config.max_active, len(active_latents))
        min_active = min(self.config.min_active, max_active)
        if min_active <= 0:
            raise ValueError("min_active must be >= 1")

        sigma = self.config.beta * (h_token.norm().item() / (d_model**0.5 + self.config.eps))

        mutations: list[SparseMutation] = []
        for _ in range(self.config.num_candidates):
            k = int(
                torch.randint(
                    min_active,
                    max_active + 1,
                    (1,),
                    generator=self.generator,
                ).item()
            )
            perm = torch.randperm(len(active_latents), generator=self.generator)[:k].tolist()
            latent_indices = [active_latents[i] for i in perm]
            coefficients = torch.randn(k, generator=self.generator, device=device, dtype=dtype) * sigma

            delta = torch.zeros(d_model, device=device, dtype=dtype)
            for alpha, latent_idx in zip(coefficients, latent_indices):
                direction = _build_decoder_direction(decoder_weight, latent_idx)
                delta = delta + (alpha * direction)

            mutations.append(
                SparseMutation(
                    latent_indices=latent_indices,
                    coefficients=coefficients,
                    delta=delta,
                )
            )

        return mutations
