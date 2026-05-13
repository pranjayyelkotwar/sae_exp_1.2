from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from search.mutations import SparseMutation


@dataclass
class TrajectoryEvent:
    step: int
    grounding_score: float
    active_latents: list[int]
    hidden_state: torch.Tensor
    mutation: SparseMutation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self.step,
            "grounding_score": float(self.grounding_score),
            "active_latents": self.active_latents,
            "hidden_state": self.hidden_state.detach().cpu().tolist(),
        }
        if self.mutation is not None:
            payload["mutation"] = {
                "latent_indices": self.mutation.latent_indices,
                "coefficients": self.mutation.coefficients.detach().cpu().tolist(),
                "delta": self.mutation.delta.detach().cpu().tolist(),
                "parent_score": self.mutation.parent_score,
                "child_score": self.mutation.child_score,
            }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class TrajectoryHistory:
    trajectory_id: str
    events: list[TrajectoryEvent] = field(default_factory=list)

    def add(self, event: TrajectoryEvent) -> None:
        self.events.append(event)

    def to_jsonl_rows(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]
