# Implementation Plan — `G_pseudo_curv`

The core idea:

Instead of explicitly computing Hessians or curvature tensors, estimate:

> “How unstable is the predictive distribution under tiny local latent perturbations?”

This gives you a stochastic proxy for local predictive curvature / brittleness.

---

# 1. Core Definition

Let:

* (h \in \mathbb{R}^d) = latent state at layer (\ell)
* (p_\theta(\cdot|h)) = next-token distribution
* (\delta) = small latent perturbation

Define:

[
G_{\text{pseudo-curv}}(h)
=========================

*

\mathbb{E}*{\delta \sim \mathcal D}
\left[
D*{KL}(p_\theta(\cdot|h)|p_\theta(\cdot|h+\delta))
\right]
]

Interpretation:

* low KL under perturbation → smooth local basin → grounded
* high KL under perturbation → brittle predictive geometry

Negative sign keeps consistency with:

* higher (G) = more grounded
* lower (G) = more de-grounded

---

# 2. Perturbation Distribution

Do NOT use isotropic Gaussian over full latent space.

That destroys locality and injects nonsense directions.

Use sparse active semantic directions only.

---

## Recommended perturbation form

At iteration (t):

[
J_t = \text{TopK}(z_t, K)
]

Sample perturbation:

[
\delta
======

\sum_{j\in S_t}
\alpha_j d_j
]

where:

* (S_t \subseteq J_t)
* (|S_t| \in [1,3])
* (d_j) = active SAE decoder directions
* (\alpha_j \sim \mathcal N(0,\sigma^2))

This keeps perturbations:

* local
* semantically meaningful
* aligned with current computation

---

# 3. Efficient Forward Pipeline

This is critical.

---

## One-time prompt forward

For prompt (q_0):

```python
with torch.no_grad():
    cache = model.run_until_layer(q0, layer=l)
```

Store:

* KV cache
* residual state (h_\ell)

---

## Perturbed continuation

For each mutation:

```python
h_perturbed = h + delta

logits = model.forward_from_layer(
    h_perturbed,
    start_layer=l
)
```

DO NOT rerun lower transformer.

This optimization is mandatory.

---

# 4. Predictive Distribution Approximation

Do NOT use full vocab softmax.

Too expensive.
Too noisy.

---

## Recommended

Restrict to:

* top-k logits
* nucleus mass

Example:

```python
topk_logits, topk_idx = logits.topk(50)
```

Then compute softmax only there.

This massively reduces:

* memory
* KL cost
* numerical instability

while preserving local geometry.

---

# 5. KL Computation

Base distribution:

[
p = p_\theta(\cdot|h)
]

Perturbed:

[
q = p_\theta(\cdot|h+\delta)
]

Compute:

[
D_{KL}(p|q)
===========

\sum_i p_i \log \frac{p_i}{q_i}
]

Implementation:

* use log-softmax
* clamp probabilities
* fp32 for stability

---

# 6. Monte Carlo Estimate

Single perturbation is noisy.

Use tiny MC estimate.

---

## Recommended

Per latent state:

```python
N = 4
```

Then:

[
G_{\text{pseudo-curv}}(h)
=========================

*

\frac1N
\sum_{i=1}^N
D_{KL}(p(h)|p(h+\delta_i))
]

This is already surprisingly stable.

---

# 7. Normalization (Important)

Raw KL explodes in high-entropy states.

Normalize.

---

## Recommended normalization

[
\tilde G_{\text{pseudo-curv}}
=============================

*

\frac{
\mathbb E[D_{KL}]
}{
H(p_\theta(\cdot|h))+\epsilon
}
]

This prevents:

* naturally uncertain states
  from dominating curvature estimates.

This normalization idea is genuinely important.

---

# 8. Evolutionary Search Integration

This is where efficiency matters most.

---

## Tiered evaluation

### Tier 1 — Cheap scoring (all candidates)

Compute:

* `G_sae`
* cheap stability proxy

No pseudo-curvature.

---

### Tier 2 — Elite candidates only

Top 10–20%.

Compute:

* MC pseudo-curvature

This cuts cost massively.

---

# 9. Curvature Cache

Very important.

Local geometry changes slowly.

Cache:

```python
curvature_cache[
    hash(active_sparse_basis)
]
```

Reuse:

* KL estimates
* perturbation statistics

until active sparse basis changes substantially.

---

# 10. Adaptive Noise Scale

Fixed (\sigma) is dangerous.

Too small:

* numerical noise

Too large:

* leaves local basin

---

## Better

Scale by latent norm:

[
\sigma_h
========

\beta \frac{|h|_2}{\sqrt d}
]

Typical:

```python
beta = 0.01
```

---

# 11. Practical Complexity

Suppose:

* population = 128
* elite fraction = 0.1
* MC samples = 4

Then:

* only ~13 candidates need pseudo-curvature
* only 52 perturbed forwards / generation

This is actually manageable.

Especially with:

* cached lower layers
* top-k vocab restriction
* batched continuation passes

---

# 12. Batched Perturbation Trick

Very important optimization.

Instead of:

```python
for delta in deltas:
    forward()
```

Stack perturbations:

```python
H_batch = h.unsqueeze(0) + delta_batch
```

Run all perturbations together.

Transformer continuation parallelizes beautifully here.

Huge throughput gain.

---

# 13. Failure Modes To Watch

## A. LayerNorm artifacts

Tiny latent perturbations can amplify weirdly after LN.

Mitigation:

* perturb post-LN residual stream consistently

---

## B. Sampling noise

Use deterministic logits.
No decoding.

---

## C. Drift collapse

Repeated evolution may move outside SAE manifold.

Mitigation:
projection regularizer:

[
R(h)=|h-D(E(h))|^2
]

Very important honestly.

---

# 14. Final Recommended Objective

I think your strongest practical formulation is:

[
G(h)
====

w_1 G_{\text{sae}}
+
w_2 G_{\text{stab-lite}}
+
w_3 G_{\text{pseudo-curv}}
--------------------------

\lambda R(h)
]

where:

* `G_sae` = semantic grounding
* `G_stab-lite` = first-order local robustness
* `G_pseudo-curv` = stochastic predictive brittleness
* `R(h)` = manifold consistency penalty

This is:

* computationally tractable
* mechanistically meaningful
* evolutionary-search compatible
* sparse-local
* scalable to long searches

without touching actual Hessians.
