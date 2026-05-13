from __future__ import annotations

from dataclasses import dataclass

import torch

from grounding_functions.curv import PseudoCurvConfig, compute_pseudo_curv
from grounding_functions.stability import StabilityConfig, compute_fisher_diag, score_stability_delta
from utils.grounding_scores import GroundingScoreCalculator


@dataclass
class GroundingWeights:
    sae: float = 1.0
    stability: float = 1.0
    curvature: float = 1.0


class GroundingEvaluator:
    def __init__(
        self,
        grounding_calc: GroundingScoreCalculator,
        weights: GroundingWeights | None = None,
        stability_config: StabilityConfig | None = None,
        curv_config: PseudoCurvConfig | None = None,
    ) -> None:
        self.grounding_calc = grounding_calc
        self.weights = weights or GroundingWeights()
        self.stability_config = stability_config or StabilityConfig()
        self.curv_config = curv_config or PseudoCurvConfig()

    def compute_sae(self, h_sparse: torch.Tensor) -> torch.Tensor:
        return self.grounding_calc.score(h_sparse)

    def compute_stability(
        self,
        delta: torch.Tensor,
        fisher_diag: torch.Tensor,
    ) -> torch.Tensor:
        return score_stability_delta(delta, fisher_diag)

    def compute_fisher_diag(
        self,
        model,
        prompt_tokens: list[int],
        override_layer: int,
        override_activations: torch.Tensor,
        token_pos: int,
    ) -> torch.Tensor:
        return compute_fisher_diag(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=override_layer,
            override_activations=override_activations,
            token_pos=token_pos,
            config=self.stability_config,
        )

    def compute_curvature(
        self,
        model,
        prompt_tokens: list[int],
        override_layer: int,
        override_activations: torch.Tensor,
        token_pos: int,
        h_dense: torch.Tensor,
        decoder_weight: torch.Tensor,
    ) -> torch.Tensor:
        return compute_pseudo_curv(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=override_layer,
            base_override_activations=override_activations,
            token_pos=token_pos,
            h_dense=h_dense,
            decoder_weight=decoder_weight,
            config=self.curv_config,
        )

    def total(
        self,
        g_sae: torch.Tensor,
        g_stab: torch.Tensor,
        g_curv: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.weights.sae * g_sae
            + self.weights.stability * g_stab
            + self.weights.curvature * g_curv
        )
