import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from sae import load_sae_model


def normalize_dataset_name(source_dataset: str) -> str:
    """Normalize dataset names from metadata to readable format."""
    if "arc" in source_dataset.lower():
        return "ARC-Easy"
    elif "hle" in source_dataset.lower():
        return "HLE"
    elif "mmlu" in source_dataset.lower():
        return "MMLU"
    else:
        return source_dataset


def load_dataset_index_mapping(metadata_file: Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    """
    Load metadata file and create mappings from activation index to dataset name.
    
    :param metadata_file: path to metadata_rank0.jsonl
    :return: (idx_to_dataset dict, dataset_to_indices dict)
    """
    logging.info(f"Loading metadata from: {metadata_file}")
    
    idx_to_dataset = {}
    dataset_to_indices = defaultdict(list)
    
    with open(metadata_file, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                record = json.loads(line)
                source_dataset = record.get('source_dataset', 'unknown')
                dataset_name = normalize_dataset_name(source_dataset)
                
                idx_to_dataset[line_idx] = dataset_name
                dataset_to_indices[dataset_name].append(line_idx)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse line {line_idx} in metadata file")
                continue
    
    logging.info(f"Loaded metadata with {len(idx_to_dataset)} activation indices")
    for dataset_name, indices in dataset_to_indices.items():
        logging.info(f"  {dataset_name}: {len(indices)} activations")
    
    return idx_to_dataset, dict(dataset_to_indices)


def iter_metadata_records(metadata_file: Path) -> tuple[str, Path, int]:
    """Yield (dataset_name, activation_path, capture_layer) from metadata lines."""
    with metadata_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse line {line_idx} in metadata file")
                continue

            source_dataset = record.get("source_dataset", "unknown")
            dataset_name = normalize_dataset_name(source_dataset)
            activation_path = record.get("activation_path")
            capture_layer = record.get("capture_layer")

            if activation_path is None or capture_layer is None:
                continue

            yield dataset_name, Path(activation_path), int(capture_layer)


def infer_batch_to_dataset_mapping(
    batch_paths: list[Path],
    idx_to_dataset: dict[int, str],
) -> dict[str, list[Path]]:
    """
    Infer which batches belong to which datasets by examining activation indices in batch files.
    
    Assumes batch files contain consecutive activation samples in order.
    
    :param batch_paths: sorted list of batch file paths
    :param idx_to_dataset: mapping from activation index to dataset name
    :return: dict mapping dataset names to lists of batch paths
    """
    logging.info("Inferring batch-to-dataset mapping from files...")
    
    batch_to_dataset = defaultdict(list)
    activation_idx = 0
    
    for batch_path in batch_paths:
        try:
            batch = torch.load(batch_path, weights_only=True)
            batch_size = batch.shape[0]
            
            # Track which datasets are in this batch
            batch_datasets = set()
            for sample_idx in range(activation_idx, activation_idx + batch_size):
                if sample_idx in idx_to_dataset:
                    batch_datasets.add(idx_to_dataset[sample_idx])
            
            # Assign batch to primary dataset (most common in this batch)
            if batch_datasets:
                primary_dataset = max(
                    batch_datasets,
                    key=lambda ds: sum(1 for i in range(activation_idx, activation_idx + batch_size)
                                     if i in idx_to_dataset and idx_to_dataset[i] == ds)
                )
                batch_to_dataset[primary_dataset].append(batch_path)
            
            activation_idx += batch_size
            
        except Exception as e:
            logging.warning(f"Error processing batch {batch_path.name}: {e}")
            continue
    
    logging.info(f"Batch-to-dataset mapping complete:")
    for dataset_name in sorted(batch_to_dataset.keys()):
        logging.info(f"  {dataset_name}: {len(batch_to_dataset[dataset_name])} batches")
    
    return batch_to_dataset


class SparseActivationAnalyzer:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        """Initialize the analyzer with SAE model."""
        self.device = device
        self.dtype = dtype
        self.model = load_sae_model(
            model_path=model_path,
            sae_top_k=8,
            sae_normalization_eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.model.eval()
        
    def analyze_batch(self, activation_batch: torch.Tensor) -> dict:
        """
        Analyze a batch of activations and extract sparse latent information.
        
        :param activation_batch: tensor of shape (batch_size, d_model)
        :return: dict with sparse latent statistics
        """
        with torch.no_grad():
            activation_batch = activation_batch.to(self.device).to(self.dtype)
            # Forward through SAE to get sparse representation
            _, h, h_sparse = self.model.forward_1d_normalized(activation_batch)
            
            # Get active latent indices (non-zero positions)
            active_latents = h_sparse.nonzero(as_tuple=True)[1]  # shape: (num_active,)
            
            return {
                'active_latents': active_latents.cpu().numpy(),
                'h_magnitude': h.abs().max().item(),
                'sparsity': (h_sparse != 0).float().mean().item(),
            }
    
    def process_dataset_batches(self, batch_paths: list[Path], dataset_name: str) -> dict:
        """
        Process all batches from a dataset and collect statistics.
        
        :param batch_paths: list of paths to batch files
        :param dataset_name: name of the dataset for logging
        :return: aggregated statistics
        """
        logging.info(f"Processing {len(batch_paths)} batches for {dataset_name}...")
        
        latent_frequencies = defaultdict(int)
        latent_co_occurrence = defaultdict(int)
        all_active_latents = []
        sparsity_values = []
        
        for i, batch_path in enumerate(batch_paths):
            if i % max(1, len(batch_paths) // 10) == 0:
                logging.info(f"  {dataset_name}: {i}/{len(batch_paths)} batches processed")
            
            batch = torch.load(batch_path, weights_only=True)
            stats = self.analyze_batch(batch)
            
            # Collect active latents
            active = stats['active_latents']
            all_active_latents.extend(active)
            sparsity_values.append(stats['sparsity'])
            
            # Count latent frequencies
            for latent_idx in active:
                latent_frequencies[int(latent_idx)] += 1
            
            # Count co-occurrences (latent pairs that fire together)
            if len(active) > 1:
                for j, lat1 in enumerate(active):
                    for lat2 in active[j+1:]:
                        pair = tuple(sorted([int(lat1), int(lat2)]))
                        latent_co_occurrence[pair] += 1
        
        return {
            'dataset_name': dataset_name,
            'num_batches': len(batch_paths),
            'latent_frequencies': dict(latent_frequencies),
            'latent_co_occurrence': dict(latent_co_occurrence),
            'all_active_latents': np.array(all_active_latents),
            'mean_sparsity': float(np.mean(sparsity_values)),
            'std_sparsity': float(np.std(sparsity_values)),
        }


def compute_average_latents_from_activations(
    activation_out_dir: Path,
    metadata_file: Path,
    output_dir: Path,
    model: "TopKSparseAutoencoder",
    device: torch.device,
    dtype: torch.dtype,
    layer: int,
    chunk_size: int,
) -> None:
    """Compute per-dataset average sparse latents from activation_outs files."""
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    dataset_sums: dict[str, torch.Tensor] = {}
    dataset_counts: dict[str, int] = defaultdict(int)
    missing_files = 0
    processed = 0

    with torch.no_grad():
        for dataset_name, activation_path, capture_layer in tqdm(
            iter_metadata_records(metadata_file),
            desc="Processing activations",
            unit="file",
        ):
            if capture_layer != layer:
                continue

            if not activation_path.is_absolute():
                activation_path = activation_out_dir / activation_path

            if activation_path.is_absolute() and not activation_path.exists():
                parts = list(activation_path.parts)
                if "activation_outs" in parts:
                    idx = parts.index("activation_outs")
                    activation_path = activation_out_dir.joinpath(*parts[idx + 1:])

            if not activation_path.exists():
                activation_path = activation_out_dir / f"layer_{layer}" / activation_path.name

            if not activation_path.exists():
                missing_files += 1
                continue

            activations = torch.load(activation_path, weights_only=True)

            if activations.dim() == 1:
                activations = activations.unsqueeze(0)

            activations = activations.to(device=device, dtype=dtype)
            tokens = activations.reshape(-1, activations.shape[-1])

            if dataset_name not in dataset_sums:
                dataset_sums[dataset_name] = torch.zeros(
                    model.n_latents,
                    device=device,
                    dtype=dtype,
                )

            for chunk in tokens.split(max(1, chunk_size)):
                _, _, h_sparse = model.forward_1d_normalized(chunk)
                dataset_sums[dataset_name] += h_sparse.sum(dim=0)
                dataset_counts[dataset_name] += h_sparse.shape[0]

            processed += 1
            if processed % 500 == 0:
                logging.info(f"Processed {processed} activation files...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, latent_sum in dataset_sums.items():
        count = dataset_counts[dataset_name]
        if count == 0:
            logging.warning(f"No tokens found for dataset {dataset_name}")
            continue

        avg_latents = latent_sum / count
        safe_name = dataset_name.lower().replace("-", "_").replace(" ", "_")
        out_path = output_dir / f"avg_latents_{safe_name}.pt"
        torch.save(avg_latents.cpu(), out_path)
        logging.info(f"Saved {dataset_name} average latents to: {out_path}")

    if missing_files > 0:
        logging.warning(f"Missing activation files: {missing_files}")



def create_grayscale_activation_heatmap(dataset_stats: list[dict], output_dir: Path) -> None:
    """
    Create a 1D grayscale heatmap showing activation patterns across latents for each dataset.
    This helps visualize clusters and patterns in latent activations.
    """
    # Get all unique latent indices across all datasets
    all_latents = set()
    for stats in dataset_stats:
        all_latents.update(stats['latent_frequencies'].keys())
    
    all_latents = sorted(all_latents)
    num_latents = len(all_latents)
    
    # Create a matrix: rows = datasets, columns = latents
    num_datasets = len(dataset_stats)
    activation_matrix = np.zeros((num_datasets, num_latents))
    
    for row_idx, stats in enumerate(dataset_stats):
        latent_freqs = stats['latent_frequencies']
        for col_idx, latent_idx in enumerate(all_latents):
            activation_matrix[row_idx, col_idx] = latent_freqs.get(latent_idx, 0)
    
    # Normalize each row for better contrast
    row_maxes = activation_matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1  # Avoid division by zero
    normalized_matrix = activation_matrix / row_maxes
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Use grayscale colormap (white = high activation, black = low/none)
    im = ax.imshow(
        normalized_matrix,
        aspect='auto',
        cmap='gray_r',  # reversed grayscale (black to white)
        interpolation='nearest'
    )
    
    # Set y-axis to dataset names
    ax.set_yticks(range(num_datasets))
    ax.set_yticklabels([stats['dataset_name'] for stats in dataset_stats], fontsize=12)
    
    # Set x-axis labels (every Nth latent to avoid crowding)
    step = max(1, num_latents // 50)  # Show ~50 tick labels
    ax.set_xticks(range(0, num_latents, step))
    ax.set_xticklabels([str(all_latents[i]) for i in range(0, num_latents, step)], 
                        fontsize=9, rotation=45)
    
    ax.set_xlabel("Latent Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dataset", fontsize=12, fontweight='bold')
    ax.set_title("1D Activation Pattern Heatmap (Grayscale)\nWhiter = Higher Activation Frequency", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Normalized Activation Frequency')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "00_grayscale_activation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 00_grayscale_activation_heatmap.png")
    
    # Also save a high-resolution version for detailed inspection
    fig, ax = plt.subplots(figsize=(40, 6))
    im = ax.imshow(
        normalized_matrix,
        aspect='auto',
        cmap='gray_r',
        interpolation='nearest'
    )
    
    ax.set_yticks(range(num_datasets))
    ax.set_yticklabels([stats['dataset_name'] for stats in dataset_stats], fontsize=14)
    
    # More granular x-axis for high-res version
    step_hires = max(1, num_latents // 200)
    ax.set_xticks(range(0, num_latents, step_hires))
    ax.set_xticklabels([str(all_latents[i]) for i in range(0, num_latents, step_hires)], 
                        fontsize=8, rotation=45)
    
    ax.set_xlabel("Latent Index", fontsize=14, fontweight='bold')
    ax.set_ylabel("Dataset", fontsize=14, fontweight='bold')
    ax.set_title("1D Activation Pattern Heatmap - High Resolution", 
                 fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, label='Normalized Activation Frequency')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "00_grayscale_activation_heatmap_hires.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 00_grayscale_activation_heatmap_hires.png")
    
    # Save detailed statistics about the heatmap
    heatmap_stats_file = output_dir / "heatmap_cluster_analysis.txt"
    with open(heatmap_stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GRAYSCALE HEATMAP CLUSTER ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Interpretation Guide:\n")
        f.write("  - White regions = High activation frequency for that dataset/latent\n")
        f.write("  - Black regions = Low/no activation frequency\n")
        f.write("  - Horizontal bands = Dataset-specific activation patterns\n")
        f.write("  - Vertical bands = Latents that activate across multiple datasets\n\n")
        
        for dataset_idx, stats in enumerate(dataset_stats):
            f.write(f"{stats['dataset_name']}:\n")
            f.write(f"  Total active latents: {len(stats['latent_frequencies'])}\n")
            
            latent_freqs = sorted(stats['latent_frequencies'].items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"  Top 10 hotspots:\n")
            for rank, (latent_idx, freq) in enumerate(latent_freqs[:10], 1):
                col_idx = all_latents.index(latent_idx)
                f.write(f"    {rank:2d}. Latent {latent_idx:5d} (col {col_idx:5d}): {freq:6d} activations\n")
            
            f.write(f"\n  Inactive latent ranges:\n")
            inactive_ranges = []
            last_active = -2
            for latent_idx in all_latents:
                if latent_idx not in stats['latent_frequencies']:
                    if latent_idx != last_active + 1:
                        if inactive_ranges and inactive_ranges[-1][1] == last_active:
                            inactive_ranges[-1] = (inactive_ranges[-1][0], latent_idx)
                        else:
                            inactive_ranges.append((latent_idx, latent_idx))
                    last_active = latent_idx
            
            for start, end in inactive_ranges[:5]:
                if start == end:
                    f.write(f"    Latent {start}\n")
                else:
                    f.write(f"    Latents {start}-{end}\n")
            f.write("\n")
    
    logging.info(f"Saved: heatmap_cluster_analysis.txt")


def create_comparison_visualizations(dataset_stats: list[dict], output_dir: Path) -> None:
    """Create comparative visualizations of sparse activations across datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. 1D Grayscale activation heatmap (showing clusters)
    create_grayscale_activation_heatmap(dataset_stats, output_dir)
    
    # 1. Side-by-side latent frequency histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Latent Activation Frequencies by Dataset", fontsize=16, fontweight='bold')
    
    for ax, stats in zip(axes, dataset_stats):
        latent_freqs = stats['latent_frequencies']
        latents = sorted(latent_freqs.keys())
        frequencies = [latent_freqs[l] for l in latents]
        
        ax.bar(latents, frequencies, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f"{stats['dataset_name']}\n(n_batches={stats['num_batches']})")
        ax.set_xlabel("Latent Index")
        ax.set_ylabel("Activation Frequency")
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_latent_frequency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 01_latent_frequency_comparison.png")
    
    # 2. Sparsity levels comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    dataset_names = [s['dataset_name'] for s in dataset_stats]
    mean_sparsities = [s['mean_sparsity'] * 100 for s in dataset_stats]
    std_sparsities = [s['std_sparsity'] * 100 for s in dataset_stats]
    
    bars = ax.bar(dataset_names, mean_sparsities, yerr=std_sparsities, capsize=10, 
                  alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=2)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("Sparsity Levels Across Datasets", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, mean_sparsities)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_sparsity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 02_sparsity_comparison.png")
    
    # 3. Top-10 active latents per dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Top-10 Most Active Latents by Dataset", fontsize=16, fontweight='bold')
    
    for ax, stats in zip(axes, dataset_stats):
        latent_freqs = stats['latent_frequencies']
        top_latents = sorted(latent_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
        latents = [str(l[0]) for l in top_latents]
        frequencies = [l[1] for l in top_latents]
        
        ax.barh(latents, frequencies, alpha=0.7, color='coral', edgecolor='black')
        ax.set_title(f"{stats['dataset_name']}")
        ax.set_xlabel("Activation Count")
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_top_latents_per_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 03_top_latents_per_dataset.png")
    
    # 4. Unique vs shared active latents (Venn-like)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latent_sets = [set(s['latent_frequencies'].keys()) for s in dataset_stats]
    dataset_names = [s['dataset_name'] for s in dataset_stats]
    
    # Calculate overlaps
    unique_counts = [len(latent_sets[i] - latent_sets[1-i] - latent_sets[2-i]) if len(latent_sets) > 2 
                     else len(latent_sets[i]) for i in range(len(latent_sets))]
    
    pairwise_overlaps = []
    for i in range(len(latent_sets)):
        for j in range(i+1, len(latent_sets)):
            overlap = len(latent_sets[i] & latent_sets[j])
            pairwise_overlaps.append(overlap)
    
    all_latents = set()
    for s in latent_sets:
        all_latents.update(s)
    
    x_pos = np.arange(len(dataset_names) + 1)
    values = unique_counts + [len(all_latents)]
    labels = dataset_names + ["Total Unique"]
    
    bars = ax.bar(x_pos, values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Unique Latents", fontsize=12)
    ax.set_title("Unique vs Shared Latent Activations", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_unique_vs_shared_latents.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 04_unique_vs_shared_latents.png")
    
    # 5. Distribution of activation frequencies (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    freq_distributions = []
    for stats in dataset_stats:
        frequencies = list(stats['latent_frequencies'].values())
        freq_distributions.append(frequencies)
    
    bp = ax.boxplot(freq_distributions, labels=dataset_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Activation Frequency per Latent", fontsize=12)
    ax.set_title("Distribution of Latent Activation Frequencies", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_frequency_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 05_frequency_distribution.png")
    
    # 6. Summary statistics table as text
    summary_file = output_dir / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPARSE ACTIVATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in dataset_stats:
            f.write(f"Dataset: {stats['dataset_name']}\n")
            f.write(f"  Number of batches: {stats['num_batches']}\n")
            f.write(f"  Mean sparsity: {stats['mean_sparsity']*100:.2f}% ± {stats['std_sparsity']*100:.2f}%\n")
            f.write(f"  Unique active latents: {len(stats['latent_frequencies'])}\n")
            
            top_5 = sorted(stats['latent_frequencies'].items(), key=lambda x: x[1], reverse=True)[:5]
            f.write(f"  Top-5 active latents: {top_5}\n\n")
        
        f.write("\nLatent Overlap Analysis:\n")
        latent_sets = [set(s['latent_frequencies'].keys()) for s in dataset_stats]
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names[i+1:], start=i+1):
                overlap = len(latent_sets[i] & latent_sets[j])
                union = len(latent_sets[i] | latent_sets[j])
                jaccard = overlap / union if union > 0 else 0
                f.write(f"  {name1} ∩ {name2}: {overlap} latents (Jaccard: {jaccard:.3f})\n")
    
    logging.info(f"Saved: summary_statistics.txt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sparse activations from SAE across datasets")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to SAE model checkpoint")
    parser.add_argument("--preprocess_dir", type=Path, default=None, help="Path to preprocessed activations directory")
    parser.add_argument("--activation_out_dir", type=Path, default=None, help="Path to activation_outs directory")
    parser.add_argument("--metadata_file", type=Path, default=None, help="Path to metadata_rank0.jsonl for dataset mapping")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--num_batches_per_dataset", type=int, default=-1, help="Number of batches to process (-1 for all)")
    parser.add_argument("--activation_layer", type=int, default=22, help="Layer index to analyze from activation_outs")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Token chunk size for SAE inference")
    parser.add_argument(
        "--average_latents_only",
        action="store_true",
        help="Compute per-dataset average latents from activation_outs and skip plots",
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

    if args.preprocess_dir is not None:
        args.preprocess_dir = args.preprocess_dir.resolve()

    if args.activation_out_dir is not None:
        args.activation_out_dir = args.activation_out_dir.resolve()
    
    if args.output_dir is None:
        if args.preprocess_dir is not None:
            args.output_dir = args.preprocess_dir.parent / "sparse_activation_analysis"
        elif args.activation_out_dir is not None:
            args.output_dir = args.activation_out_dir.parent / "sparse_activation_analysis"
        else:
            args.output_dir = Path("sparse_activation_analysis")
    args.output_dir = args.output_dir.resolve()
    
    # Resolve metadata file path
    if args.metadata_file is None:
        if args.preprocess_dir is not None:
            args.metadata_file = args.preprocess_dir.parent / "activation_outs" / "metadata_rank0.jsonl"
        elif args.activation_out_dir is not None:
            args.metadata_file = args.activation_out_dir / "metadata_rank0.jsonl"
        else:
            args.metadata_file = Path("activation_outs/metadata_rank0.jsonl")
    args.metadata_file = args.metadata_file.resolve()
    
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    device = torch.device(args.device)
    
    logging.info("=" * 80)
    logging.info("SPARSE ACTIVATION ANALYSIS WITH METADATA")
    logging.info("=" * 80)
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Preprocessed data dir: {args.preprocess_dir}")
    logging.info(f"Activation out dir: {args.activation_out_dir}")
    logging.info(f"Metadata file: {args.metadata_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Device: {device}, dtype: {dtype}")
    
    # Initialize analyzer
    analyzer = SparseActivationAnalyzer(args.model_path, device, dtype)
    
    if args.average_latents_only:
        if args.activation_out_dir is None:
            raise ValueError("--activation_out_dir is required with --average_latents_only")

        logging.info("Computing per-dataset average latents from activation_outs...")
        compute_average_latents_from_activations(
            activation_out_dir=args.activation_out_dir,
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            model=analyzer.model,
            device=device,
            dtype=dtype,
            layer=args.activation_layer,
            chunk_size=args.chunk_size,
        )
    else:
        if args.preprocess_dir is None:
            raise ValueError("--preprocess_dir is required unless --average_latents_only is set")

        # Load and process datasets using metadata
        batch_files = sorted(args.preprocess_dir.glob("batch_*.pt"))
        logging.info(f"Found {len(batch_files)} total batch files")

        # Load metadata and create dataset mappings
        if args.metadata_file.exists():
            idx_to_dataset, dataset_to_indices = load_dataset_index_mapping(args.metadata_file)
            batch_to_dataset = infer_batch_to_dataset_mapping(batch_files, idx_to_dataset)
        else:
            logging.warning(f"Metadata file not found at {args.metadata_file}")
            logging.warning("Falling back to naive batch splitting (1/3 per dataset)")
            
            # Fallback to equal split
            split_point_1 = len(batch_files) // 3
            split_point_2 = 2 * len(batch_files) // 3
            batch_to_dataset = {
                'HLE': batch_files[:split_point_1],
                'ARC-Easy': batch_files[split_point_1:split_point_2],
                'MMLU': batch_files[split_point_2:],
            }

        dataset_stats = []
        for dataset_name in sorted(batch_to_dataset.keys()):
            batches = batch_to_dataset[dataset_name]
            
            if args.num_batches_per_dataset > 0:
                batches = batches[:args.num_batches_per_dataset]
            
            if len(batches) > 0:
                stats = analyzer.process_dataset_batches(batches, dataset_name)
                dataset_stats.append(stats)
                logging.info(f"{dataset_name}: {len(stats['latent_frequencies'])} unique active latents")

        # Sort by dataset name for consistent visualization order
        dataset_stats = sorted(dataset_stats, key=lambda s: s['dataset_name'])

        # Create visualizations
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Creating comparison visualizations...")
        create_comparison_visualizations(dataset_stats, args.output_dir)
    
    logging.info("=" * 80)
    logging.info("Analysis complete! Check output directory for visualizations.")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
