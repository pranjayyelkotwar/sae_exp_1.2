from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from search.evaluator import GroundingEvaluator
from search.history import TrajectoryEvent, TrajectoryHistory
from search.mutations import SparseMutation
from search.sampler import SparseMutationSampler
from search.state import SearchState
from search.stopping import StoppingConfig, StoppingState


@dataclass
class SearchConfig:
    active_topk: int = 8


class ISLDEvolutionarySearch:
    def __init__(
        self,
        model,
        sae,
        evaluator: GroundingEvaluator,
        sampler: SparseMutationSampler,
        config: SearchConfig | None = None,
        stopping_config: StoppingConfig | None = None,
    ) -> None:
        self.model = model
        self.sae = sae
        self.evaluator = evaluator
        self.sampler = sampler
        self.config = config or SearchConfig()
        self.stopping_config = stopping_config or StoppingConfig()

    def run(self, state: SearchState) -> tuple[SearchState, TrajectoryHistory]:
        history = TrajectoryHistory(trajectory_id=state.trajectory_id)

        base_score = state.grounding_score
        stop_state = StoppingState(best_score=base_score)
        history.add(
            TrajectoryEvent(
                step=state.step,
                grounding_score=base_score,
                active_latents=[],
                hidden_state=state.hidden_state,
                mutation=None,
                metadata={"stage": "init"},
            )
        )

        iter_range = range(self.stopping_config.max_iters)
        if tqdm is not None:
            iter_range = tqdm(iter_range, desc="ISLD iterations")

        for _ in iter_range:
            if self._should_stop(state, stop_state):
                break

            base_score = state.grounding_score

            h_dense, h_sparse = self._encode_hidden(state.hidden_state)
            active_latents = self._select_active_latents(h_dense)
            if not active_latents:
                break

            fisher_diag = self.evaluator.compute_fisher_diag(
                model=self.model,
                prompt_tokens=state.prompt_tokens,
                override_layer=state.override_layer,
                override_activations=state.override_activations,
                token_pos=state.token_pos,
            )

            mutations = self.sampler.sample(
                active_latents=active_latents,
                decoder_weight=self.sae.decoder.weight,
                h_token=state.hidden_state,
            )

            eval_range = mutations
            if tqdm is not None:
                eval_range = tqdm(mutations, desc="Evaluating candidates", leave=False)

            best_mutation: SparseMutation | None = None
            best_score = None
            best_hidden = None
            best_override = None
            best_components: dict[str, Any] = {}

            for mutation in eval_range:
                hidden_state = state.hidden_state + mutation.delta
                override_activations = state.override_activations.clone()
                override_activations[0, state.token_pos] = hidden_state.to(
                    dtype=override_activations.dtype
                )

                h_dense_new, h_sparse_new = self._encode_hidden(hidden_state)

                g_sae = self.evaluator.compute_sae(h_sparse_new)
                g_stab = self.evaluator.compute_stability(mutation.delta, fisher_diag)
                g_curv = self.evaluator.compute_curvature(
                    model=self.model,
                    prompt_tokens=state.prompt_tokens,
                    override_layer=state.override_layer,
                    override_activations=override_activations,
                    token_pos=state.token_pos,
                    h_dense=h_dense_new,
                    decoder_weight=self.sae.decoder.weight,
                )
                total = self.evaluator.total(g_sae, g_stab, g_curv)

                mutation.parent_score = base_score
                mutation.child_score = float(total.item())

                if best_score is None or total.item() < best_score:
                    best_score = total.item()
                    best_mutation = mutation
                    best_hidden = hidden_state
                    best_override = override_activations
                    best_components = {
                        "g_sae": float(g_sae.item()),
                        "g_stab": float(g_stab.item()),
                        "g_curv": float(g_curv.item()),
                    }

            if best_mutation is None or best_hidden is None or best_override is None:
                break

            state.step += 1
            state.hidden_state = best_hidden
            state.override_activations = best_override
            state.grounding_score = float(best_score)

            history.add(
                TrajectoryEvent(
                    step=state.step,
                    grounding_score=state.grounding_score,
                    active_latents=active_latents,
                    hidden_state=best_hidden,
                    mutation=best_mutation,
                    metadata=best_components,
                )
            )

            if best_score < stop_state.best_score - self.stopping_config.min_improvement:
                stop_state.best_score = best_score
                stop_state.steps_since_improve = 0
            else:
                stop_state.steps_since_improve += 1

        return state, history

    def _encode_hidden(self, h_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_dense, h_sparse = self.sae.forward_1d_normalized(h_token.unsqueeze(0))
        return h_dense, h_sparse

    def _select_active_latents(self, h_dense: torch.Tensor) -> list[int]:
        k = min(self.config.active_topk, h_dense.shape[-1])
        return torch.topk(h_dense.abs().squeeze(0), k=k).indices.tolist()

    def _should_stop(self, state: SearchState, stop_state: StoppingState) -> bool:
        if self.stopping_config.target_grounding is not None:
            if state.grounding_score <= self.stopping_config.target_grounding:
                return True
        if stop_state.steps_since_improve >= self.stopping_config.patience:
            return True
        if state.step >= self.stopping_config.max_iters:
            return True
        return False
