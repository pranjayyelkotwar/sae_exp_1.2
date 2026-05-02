# Activation analysis notes

This note describes what the two analysis scripts do in this workspace.

## analyze_sparse_activations.py

- Loads a trained SAE model and runs it over preprocessed activation batches.
- Uses metadata (activation index -> dataset name) to group batches by dataset.
- Computes sparse-latent statistics per dataset (active latent IDs, sparsity, co-occurrence counts).
- Produces multiple comparison plots (heatmaps, histograms, sparsity bars, top-latent charts, overlaps, box plots).
- Writes summary text files with dataset and overlap statistics.

## capture_top_activations_pj.py

- Loads a trained SAE model and per-prompt activation tensors.
- Encodes activations into sparse latents and aggregates per prompt (max over tokens).
- For each prompt, records top-K latents; for each latent, keeps top-K prompts by score.
- Computes a dataset distribution for each latent based on its top prompts.
- Writes JSON outputs: top prompts per latent and dataset distribution per latent.

## How to run

Analyze sparse activations and generate plots:

```bash
python analyze_sparse_activations.py \
	--model_path /path/to/sae_model.pth \
	--preprocess_dir /path/to/activation_tensors_batched \
	--metadata_file /path/to/activation_outs/metadata_rank0.jsonl \
	--output_dir /path/to/sparse_activation_analysis
```

Extract top-activating prompts per latent:

```bash
python capture_top_activations_pj.py \
	--model_path /path/to/sae_model.pth \
	--metadata_path /path/to/activation_outs/metadata_rank0.jsonl \
	--output_dir /path/to/feature_outputs
```

Notes:
- If `activation_path` values in the metadata include an old prefix, pass
	`--activation_path_prefix /old/prefix/to/strip/` to make the paths load correctly.
