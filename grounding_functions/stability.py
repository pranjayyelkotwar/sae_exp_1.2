from __future__ import annotations

from dataclasses import dataclass

import torch

from llama_3.model_text_only import Transformer


@dataclass
class StabilityConfig:
    topk: int = 20
    eps: float = 1e-8


def compute_fisher_diag(
    model: Transformer,
    prompt_tokens: list[int],
    override_layer: int,
    override_activations: torch.Tensor,
    token_pos: int,
    config: StabilityConfig | None = None,
) -> torch.Tensor:
    if config is None:
        config = StabilityConfig()

    if override_activations.dim() != 3:
        raise ValueError(
            "override_activations must have shape (batch, seq_len, d_model); "
            f"got {tuple(override_activations.shape)}"
        )

    batch_size, seq_len, _ = override_activations.shape
    if batch_size != 1:
        raise ValueError("compute_fisher_diag currently supports batch_size=1")

    if seq_len != len(prompt_tokens):
        raise ValueError(
            "prompt_tokens length must match override_activations seq_len; "
            f"got tokens={len(prompt_tokens)} and seq_len={seq_len}"
        )

    if token_pos < 0:
        token_pos = max(0, seq_len - 1)
    else:
        token_pos = min(token_pos, max(0, seq_len - 1))

    device = next(model.parameters()).device
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    override_activations = override_activations.detach().clone().requires_grad_(True)

    logits = model.forward_with_activation_override_grad(
        tokens,
        start_pos=0,
        override_layer=override_layer,
        override_activations=override_activations,
    )
    logits = logits[:, token_pos, :].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    k = min(config.topk, log_probs.shape[-1])
    topk_log_probs, topk_idx = torch.topk(log_probs, k=k, dim=-1)
    topk_probs = topk_log_probs.exp()

    fisher_diag = torch.zeros_like(override_activations[:, token_pos, :], dtype=torch.float32)

    for i in range(k):
        logp_i = topk_log_probs[:, i].sum()
        grad = torch.autograd.grad(logp_i, override_activations, retain_graph=True)[0]
        g_vec = grad[:, token_pos, :].float()
        fisher_diag += topk_probs[:, i].unsqueeze(-1) * (g_vec * g_vec)

    return fisher_diag.squeeze(0)


def score_stability_delta(delta: torch.Tensor, fisher_diag: torch.Tensor) -> torch.Tensor:
    return -torch.sum(fisher_diag * (delta * delta))
