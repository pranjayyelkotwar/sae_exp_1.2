import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sae import load_sae_model


WORKSPACE_REPLACEMENTS = {
    "/workspace/llama_3_interpretability_sae": "/Users/pranjayyelkotwar/Desktop/DystopianBench/SAE_experiment/llama3_interpretability_sae",
    "/workspace/llama3_interpretability_sae": "/Users/pranjayyelkotwar/Desktop/DystopianBench/SAE_experiment/llama3_interpretability_sae",
}


def rewrite_activation_path(raw_path: str) -> Path:
    for src, dst in WORKSPACE_REPLACEMENTS.items():
        if raw_path.startswith(src):
            return Path(raw_path.replace(src, dst, 1))
    return Path(raw_path)


def load_activation_tensor(activation_path: Path) -> torch.Tensor:
    activation = torch.load(activation_path, weights_only=True)
    if activation.ndim == 1:
        activation = activation.unsqueeze(0)
    return activation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect top and least activating SAE latents for toy data"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to SAE model checkpoint (.pth)",
    )
    parser.add_argument(
        "--toy_data",
        type=Path,
        default=Path(__file__).parent / "toy_data.jsonl",
        help="Path to toy_data.jsonl",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--top_k", type=int, default=10, help="Number of top/least latents to show")
    parser.add_argument(
        "--ranking",
        type=str,
        default="raw",
        choices=["raw", "abs"],
        help="Rank by raw values or absolute magnitude",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(__file__).parent / "toy_latent_report.jsonl",
        help="Path to write JSONL report",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()
    args.model_path = args.model_path.resolve()
    args.toy_data = args.toy_data.resolve()
    args.output_path = args.output_path.resolve()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    device = torch.device(args.device)

    model = load_sae_model(
        model_path=args.model_path,
        sae_top_k=8,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=dtype,
    )
    model.eval()

    if not args.toy_data.exists():
        raise FileNotFoundError(f"toy data not found: {args.toy_data}")

    with open(args.toy_data, "r") as f, open(args.output_path, "w") as out_f:
        for line_idx, line in tqdm(list(enumerate(f)), desc="Items", unit="item"):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_text = record.get("prompt_text") or record.get("question_text") or ""
            activation_raw = record.get("activation_path")
            if not activation_raw:
                logging.warning("Line %d missing activation_path", line_idx)
                continue

            activation_path = rewrite_activation_path(activation_raw)
            if not activation_path.exists():
                logging.warning("Activation file missing: %s", activation_path)
                continue

            activation = load_activation_tensor(activation_path).to(device).to(dtype)

            with torch.no_grad():
                _, h, _ = model.forward_1d_normalized(activation)

            h = h.squeeze(0)
            if h.ndim == 2:
                rank_values = h.abs() if args.ranking == "abs" else h
                score_values = rank_values.max(dim=0).values
                raw_values = h.max(dim=0).values
            else:
                score_values = h.abs() if args.ranking == "abs" else h
                raw_values = h
            top_vals, top_idx = torch.topk(
                score_values, k=min(args.top_k, score_values.numel()), largest=True
            )
            low_vals, low_idx = torch.topk(
                score_values, k=min(args.top_k, score_values.numel()), largest=False
            )

            top_latents = []
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                top_latents.append(
                    {
                        "latent": int(idx),
                        "score": float(val),
                        "raw": float(raw_values[idx].item()),
                    }
                )

            low_latents = []
            for idx, val in zip(low_idx.tolist(), low_vals.tolist()):
                low_latents.append(
                    {
                        "latent": int(idx),
                        "score": float(val),
                        "raw": float(raw_values[idx].item()),
                    }
                )

            out_record = {
                "item_index": line_idx,
                "prompt_text": prompt_text,
                "activation_path": str(activation_path),
                "ranking": args.ranking,
                "top_latents": top_latents,
                "least_latents": low_latents,
            }
            out_f.write(json.dumps(out_record) + "\n")

    logging.info("Saved report to %s", args.output_path)


if __name__ == "__main__":
    main()


# python toy_exp/inspect_toy_activations.py \
#   --model_path ../model_checkpoint_epoch-10\ \(1\).pth \
#   --ranking abs \
#   --output_path /tmp/toy_latents.jsonl