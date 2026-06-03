import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from grounding_functions.perplexity_regression import (
    PerplexityRegressionWeights,
    save_perplexity_regression_weights,
)
from sae import load_sae_model


@dataclass
class RegressionConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-2
    weight_decay: float = 0.0
    normalize_targets: bool = True
    use_bias: bool = True


def parse_activation_indices(raw: str) -> list[int]:
    if not raw:
        return []
    if raw.strip().lower() in {"all", "*"}:
        return []
    return [int(idx.strip()) for idx in raw.split(",") if idx.strip()]


def find_activation_indices(activation_dir: Path, layer: int) -> list[int]:
    layer_dir = activation_dir / f"layer_{layer}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Activation layer directory not found: {layer_dir}")

    indices: list[int] = []
    prefix = f"activations_l{layer}_idx"
    for path in layer_dir.glob(f"{prefix}*.pt"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        raw_idx = stem[len(prefix) :]
        try:
            indices.append(int(raw_idx))
        except ValueError:
            continue

    return sorted(indices)


def resolve_activation_path(activation_dir: Path, layer: int, sample_idx: int) -> Path:
    return activation_dir / f"layer_{layer}" / f"activations_l{layer}_idx{sample_idx}.pt"


def load_activation_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def choose_token_pos(token_pos: int, seq_len: int) -> int:
    if token_pos < 0:
        return max(0, seq_len - 1)
    return min(token_pos, max(0, seq_len - 1))


def load_target_map(
    perplexity_jsonl: Path,
    target_key: str,
    layer_filter: int | None,
) -> dict[int, float]:
    if not perplexity_jsonl.exists():
        raise FileNotFoundError(f"Perplexity JSONL not found: {perplexity_jsonl}")

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with perplexity_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            capture_layer = record.get("capture_layer")
            if layer_filter is not None and capture_layer is not None:
                if int(capture_layer) != layer_filter:
                    continue

            idx = record.get("capture_dataset_idx")
            if idx is None:
                continue

            target = record.get(target_key)
            if target is None:
                continue

            idx_int = int(idx)
            sums[idx_int] = sums.get(idx_int, 0.0) + float(target)
            counts[idx_int] = counts.get(idx_int, 0) + 1

    return {idx: sums[idx] / counts[idx] for idx in sums}


def train_regression(
    features: torch.Tensor,
    targets: torch.Tensor,
    config: RegressionConfig,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    if targets.numel() == 0:
        raise ValueError("No training targets provided")

    target_mean = targets.mean().item()
    target_std = targets.std(unbiased=False).item()
    target_std = target_std if target_std > 0 else 1.0

    if config.normalize_targets:
        targets = (targets - target_mean) / target_std

    model = torch.nn.Linear(features.shape[1], 1, bias=config.use_bias)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    indices = torch.arange(features.shape[0])
    epoch_iter = range(config.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="Regression epochs")

    for _ in epoch_iter:
        perm = indices[torch.randperm(indices.numel())]
        for start in range(0, perm.numel(), config.batch_size):
            batch_idx = perm[start : start + config.batch_size]
            batch_x = features[batch_idx].to(device)
            batch_y = targets[batch_idx].to(device)

            pred = model(batch_x).squeeze(-1)
            loss = torch.mean((pred - batch_y) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(features.to(device)).squeeze(-1)
        mse = torch.mean((preds - targets.to(device)) ** 2).item()

    weights = model.weight.detach().cpu().squeeze(0)
    bias = float(model.bias.detach().cpu().squeeze(0).item()) if config.use_bias else 0.0

    return weights, bias, mse, (target_std if config.normalize_targets else 1.0), target_mean


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression weights to predict perplexity from SAE latents")
    parser.add_argument("--sae_model_path", type=Path, required=True)
    parser.add_argument("--activation_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--perplexity_jsonl", type=Path, default=Path("results/grounding_perplexity.jsonl"))
    parser.add_argument("--activation_indices", type=str, default="all")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--target_key", type=str, default="perplexity", choices=["perplexity", "mean_nll", "sum_nll"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sae_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--normalize_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize targets before training (stored mean/std for de-normalization).",
    )
    parser.add_argument(
        "--use_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include bias term in regression model.",
    )
    parser.add_argument("--output_weights", type=Path, default=Path("results/perplexity_regression_weights.pt"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()
    args.activation_dir = args.activation_dir.resolve()
    args.perplexity_jsonl = args.perplexity_jsonl.resolve()
    args.output_weights = args.output_weights.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_dtype = torch.float32 if args.sae_dtype == "float32" else torch.float16

    logging.info("Loading SAE model...")
    sae = load_sae_model(
        model_path=args.sae_model_path,
        sae_top_k=8,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=sae_dtype,
    )

    target_map = load_target_map(
        perplexity_jsonl=args.perplexity_jsonl,
        target_key=args.target_key,
        layer_filter=args.layer,
    )
    if not target_map:
        raise ValueError("No target values found; check perplexity JSONL and target_key")

    activation_indices = parse_activation_indices(args.activation_indices)
    if not activation_indices:
        logging.info("No explicit indices provided; scanning activation directory...")
        activation_indices = find_activation_indices(args.activation_dir, args.layer)

    if args.max_samples is not None:
        activation_indices = activation_indices[: args.max_samples]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    features: list[torch.Tensor] = []
    targets: list[float] = []
    missing_targets = 0
    missing_activations = 0

    index_iter = activation_indices
    if tqdm is not None:
        index_iter = tqdm(index_iter, desc="Encoding activations")

    for sample_idx in index_iter:
        target = target_map.get(sample_idx)
        if target is None:
            missing_targets += 1
            continue

        activation_path = resolve_activation_path(args.activation_dir, args.layer, sample_idx)
        if not activation_path.exists():
            missing_activations += 1
            continue

        activations = load_activation_tensor(activation_path)
        if activations.dim() != 2:
            raise ValueError(
                f"Expected activations shape (seq_len, d_model), got {tuple(activations.shape)}"
            )

        seq_len, _ = activations.shape
        token_pos = choose_token_pos(args.token_pos, seq_len)
        h_token = activations[token_pos].to(device=device, dtype=sae_dtype)

        with torch.no_grad():
            _, _, h_sparse = sae.forward_1d_normalized(h_token.unsqueeze(0))

        features.append(h_sparse.squeeze(0).detach().cpu().float())
        targets.append(float(target))

    if not features:
        raise ValueError("No training samples found; check activation indices and targets")

    logging.info(
        "Training regression with %s samples (skipped %s missing targets, %s missing activations)",
        len(features),
        missing_targets,
        missing_activations,
    )

    feature_tensor = torch.stack(features, dim=0)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    config = RegressionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        normalize_targets=args.normalize_targets,
        use_bias=args.use_bias,
    )

    weights, bias, mse, target_std, target_mean = train_regression(
        features=feature_tensor,
        targets=target_tensor,
        config=config,
        device=device,
    )

    payload = PerplexityRegressionWeights(
        weights=weights,
        bias=bias,
        target_mean=target_mean,
        target_std=target_std,
        target_key=args.target_key,
        normalized=args.normalize_targets,
        layer=args.layer,
        token_pos=args.token_pos,
    )
    save_perplexity_regression_weights(payload, args.output_weights)

    logging.info("Finished training. MSE=%.6f", mse)
    logging.info("Saved regression weights to %s", args.output_weights)


if __name__ == "__main__":
    main()
