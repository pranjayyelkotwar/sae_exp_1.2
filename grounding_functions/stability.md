# Implementation Writeup: First-Order Approximation for (G_{\text{stab}})

The original Fisher-based stability score is:

[
G_{\text{stab}}(h)
==================

*

\mathbb E_{y\sim p(\cdot|h)}
\left[
|\nabla_h \log p(y|h)|_2^2
\right]
]

G_{stab}(h)=-\mathbb E_{y\sim p(\cdot\mid h)}\left[|\nabla_h\log p(y\mid h)|_2^2\right]

Directly recomputing this for every perturbed hidden state during evolutionary search is computationally expensive because each perturbation requires:

* a forward pass through the remaining transformer layers,
* logit computation,
* and backward differentiation with respect to the hidden state.

To make iterative latent search tractable, we instead use a first-order Taylor approximation around the current hidden state.

---

# 1. Local Linearization of Predictive Log-Probabilities

Let:

[
h \in \mathbb R^d
]

be the current hidden state at intervention layer (l).

For token (y), define:

[
g_y
===

\nabla_h \log p(y|h)
]

g_y=\nabla_h\log p(y\mid h)

Using a first-order Taylor expansion:

[
\log p(y|h+\delta h)
\approx
\log p(y|h)
+
g_y^\top \delta h
]

\log p(y\mid h+\delta h)\approx \log p(y\mid h)+g_y^\top\delta h

Thus, after computing gradients once at the current state (h), nearby perturbations can be evaluated without rerunning the transformer suffix.

---

# 2. Approximate Fisher Geometry

The Fisher Information Matrix at hidden state (h) is:

[
I(h)
====

\mathbb E_{y\sim p(\cdot|h)}
[g_yg_y^\top]
]

I(h)=\mathbb E_{y\sim p(\cdot\mid h)}[g_yg_y^\top]

In practice, we approximate the expectation using only the top-(K) predicted tokens:

[
I(h)
\approx
\sum_{y\in \text{TopK}}
p(y|h),
g_yg_y^\top
]

I(h)\approx \sum_{y\in TopK} p(y\mid h) g_y g_y^\top

This matrix captures the local predictive sensitivity geometry around the current hidden state.

---

# 3. Fast Perturbation Scoring

For a candidate perturbation:

[
\delta h
]

the local KL divergence induced by the perturbation is approximated by:

[
D_{KL}(p(h)|p(h+\delta h))
\approx
\frac12
\delta h^\top I(h)\delta h
]

D_{KL}(p(h)|p(h+\delta h))\approx \frac12\delta h^\top I(h)\delta h

We therefore define the approximate local stability score:

[
\widetilde G_{\text{stab}}(h,\delta h)
======================================

*

\delta h^\top I(h)\delta h
]

\widetilde G_{stab}(h,\delta h)=-\delta h^\top I(h)\delta h

Interpretation:

* large quadratic value → perturbation strongly alters predictive beliefs,
* small quadratic value → perturbation remains locally stable.

The negative sign preserves the convention that larger values correspond to greater stability.

---

# 4. Search Procedure

At iteration (t):

1. Extract hidden state:

[
h_t
]

2. Run suffix transformer once to obtain:

   * logits,
   * probabilities,
   * gradients (g_y) for top-(K) tokens.

3. Construct approximate Fisher matrix:

[
I(h_t)
]

4. Generate candidate perturbations:

[
\delta h_i
]

from:

* SAE directions,
* sparse feature combinations,
* evolutionary mutations,
* or random local search.

5. Score each perturbation using:

[
s_i
===

*

\delta h_i^\top I(h_t)\delta h_i
]

6. Choose the perturbation minimizing local grounding stability.

---

# 5. Computational Complexity

Without approximation:

* every perturbation requires forward + backward through remaining layers.

With first-order approximation:

* one forward/backward pass per search iteration,
* then all candidate evaluations reduce to quadratic forms.

This changes complexity from:

[
O(N_{\text{candidates}}\times \text{transformer cost})
]

to:

[
O(\text{transformer cost})
+
O(N_{\text{candidates}}\times d^2)
]

or lower if diagonal Fisher approximations are used.

---

# 6. Practical Simplifications

In practice:

* use only top-10 or top-20 tokens,
* optionally approximate Fisher diagonally:

[
I(h)\approx \operatorname{diag}(v)
]

yielding:

[
\delta h^\top I(h)\delta h
==========================

\sum_i v_i (\delta h_i)^2
]

which is extremely fast.

---

# 7. Recompute Strategy

The first-order approximation is only locally valid.

Therefore:

* reuse the same Fisher approximation for several small perturbations,
* recompute after:

  * large latent movement,
  * search stagnation,
  * or every (M) iterations.

This maintains local geometric fidelity while avoiding repeated full transformer evaluations.
