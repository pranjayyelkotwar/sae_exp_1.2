import argparse
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, weights_only=True)


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  mean: {tensor.mean().item():.6f}")
    print(f"  std:  {tensor.std().item():.6f}")
    print("  values:")
    print(tensor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect avg latent tensors")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("sparse_activation_analysis"),
        help="Directory containing avg_latents_*.pt",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()

    torch.set_printoptions(threshold=float("inf"))

    arc_path = input_dir / "avg_latents_arc_easy.pt"
    hle_path = input_dir / "avg_latents_hle.pt"
    mmlu_path = input_dir / "avg_latents_mmlu.pt"
    diff_path = input_dir / "grounding_weights.pt"
    arc = load_tensor(arc_path)
    hle = load_tensor(hle_path)
    mmlu = load_tensor(mmlu_path)

    print_tensor_info("ARC-Easy", arc)
    print_tensor_info("HLE", hle)
    print_tensor_info("MMLU", mmlu)

    if arc.shape != hle.shape:
        raise ValueError(f"Shape mismatch: ARC-Easy {arc.shape} vs HLE {hle.shape}")

    diff = arc - hle
    torch.save(diff, diff_path)

    print("ARC-Easy - HLE:")
    print(f"  mean: {diff.mean().item():.6f}")
    print(f"  mean abs: {diff.abs().mean().item():.6f}")
    print(f"  max abs: {diff.abs().max().item():.6f}")


if __name__ == "__main__":
    main()



# python dev/inspect_avg_latents.py --input_dir sparse_activation_analysis