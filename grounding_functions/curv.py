from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from inference_activations import logits_with_activation_override


@dataclass
class PseudoCurvConfig:
    topk_vocab: int = 50
    mc_samples: int = 4
    min_active: int = 1
    max_active: int = 3
    beta: float = 0.01
    eps: float = 1e-8
    latent_topk: int = 8
    normalize_entropy: bool = True


def _build_decoder_direction(
    decoder_weight: torch.Tensor,
    latent_idx: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    direction = decoder_weight[:, latent_idx]
    norm = direction.norm() + eps
    return direction / norm


def _sample_sparse_deltas(
    decoder_weight: torch.Tensor,
    active_latents: Iterable[int],
    num_samples: int,
    min_active: int,
    max_active: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    active_latents = list(active_latents)
    if not active_latents:
        raise ValueError("active_latents must be non-empty")

    max_active = min(max_active, len(active_latents))
    min_active = min(min_active, max_active)

    deltas = []
    for _ in range(num_samples):
        k = int(torch.randint(min_active, max_active + 1, (1,), generator=generator).item())
        perm = torch.randperm(len(active_latents), generator=generator)[:k].tolist()
        indices = [active_latents[i] for i in perm]
        alphas = torch.randn(k, generator=generator, device=device, dtype=dtype) * sigma

        delta = torch.zeros(decoder_weight.shape[0], device=device, dtype=dtype)
        for alpha, latent_idx in zip(alphas, indices):
            direction = _build_decoder_direction(decoder_weight, latent_idx)
            delta = delta + (alpha * direction)
        deltas.append(delta)

    return torch.stack(deltas, dim=0)


def compute_pseudo_curv(
    model,
    prompt_tokens: list[int],
    override_layer: int,
    base_override_activations: torch.Tensor,
    token_pos: int,
    h_dense: torch.Tensor,
    decoder_weight: torch.Tensor,
    config: PseudoCurvConfig | None = None,
) -> torch.Tensor:
    """Estimate pseudo-curvature as expected KL under sparse latent perturbations.

    Args:
        model: LLM object consumed by logits_with_activation_override.
        prompt_tokens: Tokenized prompt matching base_override_activations seq_len.
        override_layer: Layer index where activations are overridden.
        base_override_activations: Residual stream activations (batch, seq_len, d_model).
        token_pos: Sequence position to perturb; will be clamped to valid range.
        h_dense: SAE latent activations for the same token position (shape [1, d_latent]
    max_batch = config.mc_samples
    if hasattr(model, "params") and getattr(model.params, "max_batch_size", None):
        max_batch = max(1, int(model.params.max_batch_size))
    cache_limit = None
    if hasattr(model, "layers") and model.layers:
        try:
            cache_limit = model.layers[0].attention.cache_k.shape[0]
        except Exception:
            cache_limit = None
    if cache_limit is not None:
        max_batch = min(max_batch, int(cache_limit))
            or [d_latent]). Used to pick top active latents.
        decoder_weight: SAE decoder matrix (d_model, d_latent); each column is a latent
            direction in residual space.
        config: Optional PseudoCurvConfig overrides.
    """
    if config is None:
        config = PseudoCurvConfig()

    if base_override_activations.dim() != 3:
        raise ValueError(
            "base_override_activations must have shape (batch, seq_len, d_model); "
            f"got {tuple(base_override_activations.shape)}"
        )

    batch_size, seq_len, d_model = base_override_activations.shape
    if batch_size != 1:
        raise ValueError("compute_pseudo_curv currently supports batch_size=1")

    if seq_len != len(prompt_tokens):
        raise ValueError(
            "prompt_tokens length must match override_activations seq_len; "
            f"got tokens={len(prompt_tokens)} and seq_len={seq_len}"
        )

    if token_pos < 0:
        token_pos = max(0, seq_len - 1)
    else:
        token_pos = min(token_pos, max(0, seq_len - 1))

    device = base_override_activations.device
    dtype = base_override_activations.dtype

    h_token = base_override_activations[0, token_pos]
    sigma = config.beta * (h_token.norm().item() / (d_model**0.5))

    base_logits = logits_with_activation_override(
        model=model,
        prompt_tokens=prompt_tokens,
        override_layer=override_layer,
        override_activations=base_override_activations,
        token_pos=token_pos,
    )
    base_logits = base_logits.float()
    k = min(config.topk_vocab, base_logits.shape[-1])
    topk_logits, topk_idx = torch.topk(base_logits, k=k, dim=-1)

    base_log_probs = torch.log_softmax(topk_logits, dim=-1)
    base_probs = base_log_probs.exp()

    entropy = -(base_probs * base_log_probs).sum(dim=-1)

    latent_k = min(config.latent_topk, h_dense.shape[-1])
    active_latents = torch.topk(h_dense.abs().squeeze(0), k=latent_k).indices.tolist()

    deltas = _sample_sparse_deltas(
        decoder_weight=decoder_weight,
        active_latents=active_latents,
        num_samples=config.mc_samples,
        min_active=config.min_active,
        max_active=config.max_active,
        sigma=sigma,
        device=device,
        dtype=dtype,
    )

    override_batch = base_override_activations.repeat(config.mc_samples, 1, 1)
    override_batch[:, token_pos, :] = override_batch[:, token_pos, :] + deltas

    max_batch = config.mc_samples
    if hasattr(model, "params") and getattr(model.params, "max_batch_size", None):
        max_batch = max(1, int(model.params.max_batch_size))
    cache_limit = None
    if hasattr(model, "layers") and model.layers:
        try:
            cache_limit = model.layers[0].attention.cache_k.shape[0]
        except Exception:
            cache_limit = None
    if cache_limit is not None:
        max_batch = min(max_batch, int(cache_limit))

    pert_chunks = []
    for start in range(0, config.mc_samples, max_batch):
        chunk = override_batch[start : start + max_batch]
        chunk_logits = logits_with_activation_override(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=override_layer,
            override_activations=chunk,
            token_pos=token_pos,
        )
        pert_chunks.append(chunk_logits)

    pert_logits = torch.cat(pert_chunks, dim=0).float()

    topk_idx = topk_idx.repeat(config.mc_samples, 1)
    pert_topk_logits = torch.gather(pert_logits, dim=-1, index=topk_idx)
    pert_log_probs = torch.log_softmax(pert_topk_logits, dim=-1)

    kl = (base_probs.repeat(config.mc_samples, 1) * (base_log_probs.repeat(config.mc_samples, 1) - pert_log_probs)).sum(
        dim=-1
    )

    if config.normalize_entropy:
        kl = kl / (entropy.repeat(config.mc_samples) + config.eps)

    return -kl.mean()
