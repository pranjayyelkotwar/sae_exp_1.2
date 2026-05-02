import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from sae import load_sae_model


# -----------------------------
# Utils
# -----------------------------
def normalize_dataset_name(source_dataset: str) -> str:
    s = source_dataset.lower()
    if "arc" in s:
        return "ARC-Easy"
    elif "hle" in s:
        return "HLE"
    elif "mmlu" in s:
        return "MMLU"
    return source_dataset

def load_metadata(metadata_path: Path):
    records = []
    with open(metadata_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# -----------------------------
# SAE Wrapper
# -----------------------------
class SAEFeatureExplorer:
    def __init__(self, model_path: Path, device="cpu"):
        self.device = torch.device(device)

        self.model = load_sae_model(
            model_path=model_path,
            sae_top_k=8,
            sae_normalization_eps=1e-6,
            device=self.device,
            dtype=torch.float32,
        )
        self.model.eval()

    def encode(self, activations: torch.Tensor):
        """
        activations: (T, d_model)
        returns: (T, n_latents)
        """
        with torch.no_grad():
            activations = activations.to(self.device).float()
            _, _, h_sparse = self.model.forward_1d_normalized(activations)
        return h_sparse.cpu()


# -----------------------------
# Core Extraction
# -----------------------------
def extract_topk_features(
    model: SAEFeatureExplorer,
    metadata,
    top_k_per_latent=10,
    top_k_per_prompt=10,
    activation_path_prefix: str | None = None,
):
    """
    Returns:
        dict: latent_id -> list of {score, dataset, text}
    """

    latent_topk = defaultdict(list)

    for record in tqdm(metadata, desc="Processing prompts"):



        act_path = record["activation_path"]
        if activation_path_prefix and act_path.startswith(activation_path_prefix):
            act_path = act_path[len(activation_path_prefix):]
        act_path = Path(act_path)

        text = record["prompt_text"]
        dataset = normalize_dataset_name(record["source_dataset"])

        if not act_path.exists():
            continue

        # Load activations (T, d_model)
        acts = torch.load(act_path)

        # Encode → (T, n_latents)
        h_sparse = model.encode(acts)

        # Aggregate per prompt (max over tokens)
        feature_vals = h_sparse.max(dim=0).values  # (n_latents,)

        # Top features for this prompt
        top_vals, top_idx = torch.topk(feature_vals, k=top_k_per_prompt)

        for val, idx in zip(top_vals, top_idx):
            latent_id = int(idx)

            latent_topk[latent_id].append({
                "score": float(val),
                "dataset": dataset,
                "text": text,
            })

    # Keep only top-K per latent
    for latent_id in latent_topk:
        latent_topk[latent_id] = sorted(
            latent_topk[latent_id],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k_per_latent]

    return latent_topk


# -----------------------------
# Dataset Distribution
# -----------------------------
def compute_dataset_distribution(latent_topk):
    stats = {}

    for latent_id, items in latent_topk.items():
        counts = defaultdict(int)
        for item in items:
            counts[item["dataset"]] += 1

        stats[f"latent_{latent_id}"] = dict(counts)

    return stats


# -----------------------------
# Save Functions
# -----------------------------
def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract top activating prompts per SAE latent")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("feature_outputs"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k_per_latent", type=int, default=10)
    parser.add_argument("--top_k_per_prompt", type=int, default=10)
    parser.add_argument(
        "--activation_path_prefix",
        type=str,
        default=None,
        help="Prefix to strip from activation_path values before loading",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.model_path = args.model_path.resolve()
    args.metadata_path = args.metadata_path.resolve()
    args.output_dir = args.output_dir.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    explorer = SAEFeatureExplorer(args.model_path, device=args.device)

    print("Loading metadata...")
    metadata = load_metadata(args.metadata_path)

    print("Extracting top activating prompts per latent...")
    latent_topk = extract_topk_features(
        explorer,
        metadata,
        top_k_per_latent=args.top_k_per_latent,
        top_k_per_prompt=args.top_k_per_prompt,
        activation_path_prefix=args.activation_path_prefix,
    )

    print("Computing dataset distribution...")
    dataset_stats = compute_dataset_distribution(latent_topk)

    print("Saving results...")
    save_json(latent_topk, args.output_dir / "latent_top_prompts.json")
    save_json(dataset_stats, args.output_dir / "latent_dataset_distribution.json")

    print("Done.")


if __name__ == "__main__":
    main()