import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from sae import load_sae_model


def g_sae(z: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (c * z).sum(dim=-1) / (z.sum(dim=-1) + eps)


class GroundingFunction:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        layer: int,
        c_vector: torch.Tensor,
        chunk_size: int = 4096,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.layer = layer
        self.chunk_size = chunk_size
        self.c_vector = c_vector.to(device=device, dtype=dtype)
        self.model = load_sae_model(
            model_path=model_path,
            sae_top_k=8,
            sae_normalization_eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.model.eval()

    def compute_scores(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        activations = activations.to(device=self.device, dtype=self.dtype)
        tokens = activations.reshape(-1, activations.shape[-1])
        scores = []

        with torch.no_grad():
            for chunk in tokens.split(max(1, self.chunk_size)):
                _, _, h_sparse = self.model.forward_1d_normalized(chunk)
                scores.append(g_sae(h_sparse, self.c_vector))

        return torch.cat(scores, dim=0)

    def _resolve_activation_path(self, activation_out_dir: Path, activation_path: str) -> Path:
        path = Path(activation_path)
        if not path.is_absolute():
            path = activation_out_dir / path

        if path.is_absolute() and not path.exists():
            parts = list(path.parts)
            if "activation_outs" in parts:
                idx = parts.index("activation_outs")
                path = activation_out_dir.joinpath(*parts[idx + 1:])

        if not path.exists():
            path = activation_out_dir / f"layer_{self.layer}" / path.name

        return path

    def iter_activation_paths_from_metadata(
        self,
        activation_out_dir: Path,
        metadata_file: Path,
        max_files_per_dataset: int,
    ) -> list[Path]:
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        counts: dict[str, int] = defaultdict(int)
        paths: list[Path] = []

        with metadata_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                activation_path = record.get("activation_path")
                capture_layer = record.get("capture_layer")
                dataset_name = record.get("source_dataset", "unknown")

                if activation_path is None or capture_layer is None:
                    continue

                if int(capture_layer) != self.layer:
                    continue

                if max_files_per_dataset > 0 and counts[dataset_name] >= max_files_per_dataset:
                    continue

                path = self._resolve_activation_path(activation_out_dir, activation_path)
                if not path.exists():
                    continue

                paths.append(path)
                counts[dataset_name] += 1

        return paths

    def process_activation_files(
        self,
        activation_dir: Path,
        output_dir: Path,
        files: list[Path] | None = None,
    ) -> None:
        activation_dir = activation_dir / f"layer_{self.layer}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if files is None:
            files = sorted(activation_dir.glob(f"activations_l{self.layer}_idx*.pt"))

        for path in tqdm(files, desc="Grounding activations", unit="file"):
            activations = torch.load(path, weights_only=True)
            scores = self.compute_scores(activations)
            out_path = output_dir / f"grounding_scores_{path.stem}.pt"
            torch.save(scores.cpu(), out_path)

    def compute_dataset_means_from_metadata(
        self,
        activation_out_dir: Path,
        metadata_file: Path,
        max_samples_per_dataset: int,
    ) -> dict[str, float]:
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        missing_files = 0

        with metadata_file.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Dataset grounding means", unit="record"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                dataset_name = record.get("source_dataset", "unknown")
                activation_path = record.get("activation_path")
                capture_layer = record.get("capture_layer")

                if activation_path is None or capture_layer is None:
                    continue

                if int(capture_layer) != self.layer:
                    continue

                if max_samples_per_dataset > 0 and counts[dataset_name] >= max_samples_per_dataset:
                    continue

                path = self._resolve_activation_path(activation_out_dir, activation_path)

                if not path.exists():
                    missing_files += 1
                    continue

                activations = torch.load(path, weights_only=True)
                scores = self.compute_scores(activations)

                sums[dataset_name] += scores.mean().item()
                counts[dataset_name] += 1

        if missing_files > 0:
            print(f"Warning: missing activation files: {missing_files}")

        return {name: sums[name] / counts[name] for name in counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute grounding scores for activations")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--activation_out_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--avg_latents_dir", type=Path, default=Path("sparse_activation_analysis"))
    parser.add_argument("--output_dir", type=Path, default=Path("grounding_scores"))
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--metadata_file", type=Path, default=Path("activation_outs/metadata_rank0.jsonl"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--max_files_per_dataset",
        type=int,
        default=-1,
        help="Max activation files per dataset for grounding outputs (-1 for all)",
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=100,
        help="Max activation files per dataset for averaging (-1 for all)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    arc_path = args.avg_latents_dir / "avg_latents_arc_easy.pt"
    hle_path = args.avg_latents_dir / "avg_latents_hle.pt"

    arc = torch.load(arc_path, weights_only=True)
    hle = torch.load(hle_path, weights_only=True)

    if arc.shape != hle.shape:
        raise ValueError(f"Shape mismatch: ARC-Easy {arc.shape} vs HLE {hle.shape}")

    c_vector = arc - hle

    gf = GroundingFunction(
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        layer=args.layer,
        c_vector=c_vector,
        chunk_size=args.chunk_size,
    )
    files = None
    if args.max_files_per_dataset > 0:
        files = gf.iter_activation_paths_from_metadata(
            activation_out_dir=args.activation_out_dir,
            metadata_file=args.metadata_file,
            max_files_per_dataset=args.max_files_per_dataset,
        )

    gf.process_activation_files(args.activation_out_dir, args.output_dir, files=files)

    dataset_means = gf.compute_dataset_means_from_metadata(
        activation_out_dir=args.activation_out_dir,
        metadata_file=args.metadata_file,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )

    print("\nAverage grounding score by dataset:")
    for name in sorted(dataset_means.keys()):
        print(f"  {name}: {dataset_means[name]:.6f}")


if __name__ == "__main__":
    main()
