from pathlib import Path
import json
import torch
from grounding_functions.perplexity_regression import load_perplexity_regression_weights

cache_path = Path("results/perplexity_regression_cache_l22_t-1_perplexity.pt")
weights_path = Path("results/perplexity_regression_weights.pt")

payload = torch.load(cache_path, map_location="cpu")
features = payload["features"]
targets = payload["targets"]

weights = load_perplexity_regression_weights(
    weights_path,
    device=torch.device("cpu"),
    dtype=features.dtype,
)

preds = weights.predict(features)
abs_err = (preds - targets).abs()
tol = 0.25 * targets.abs()
acc = (abs_err <= tol).float().mean().item()

w = weights.weights.detach().cpu()
def q(p): return torch.quantile(w, torch.tensor(p)).item()

print(f"n={w.numel()}, mean={w.mean():.6f}, std={w.std(unbiased=False):.6f}, min={w.min():.6f}, max={w.max():.6f}")
print(f"p01={q(0.01):.6f}, p05={q(0.05):.6f}, p25={q(0.25):.6f}, p50={q(0.50):.6f}, p75={q(0.75):.6f}, p95={q(0.95):.6f}, p99={q(0.99):.6f}")
print(f"pos_frac={(w>0).float().mean().item():.4f}, near_zero_frac={(w.abs()<1e-8).float().mean().item():.4f}")
print(f"accuracy@25%_relative={acc:.6f}")