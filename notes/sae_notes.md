# SAE pipeline notes

This note summarizes the purpose and flow of the SAE preprocessing, training, and model code.

## sae_preprocessing.py

- Scans a directory of activation tensors and batches them into fixed-size chunks.
- Computes the dataset mean activation vector using a streaming Welford-style accumulator.
- Saves batched activation tensors to disk and writes the mean tensor for training.
- Uses multiprocessing to parallelize preprocessing over multiple files.

## sae_training.py

- Trains a Top-K Sparse Autoencoder (SAE) with distributed data parallel (DDP).
- Loads the precomputed mean vector (`b_pre`) and initializes the SAE model.
- Builds train/validation splits from pre-batched activation files.
- Uses an auxiliary loss to revive dead latents, with logging to Weights & Biases.
- Saves checkpoints per epoch and exports the final trained model.

## How to run

Preprocess activations into fixed-size batches and compute the mean vector:

```bash
python sae_preprocessing.py \
	--input_dir /path/to/activation_tensors \
	--num_processes 8 \
	--batch_size 1024
```

Train with DDP using `torchrun` (one process per GPU):

```bash
torchrun --nproc_per_node 8 sae_training.py \
	--data_dir /path/to/activation_tensors_batched \
	--b_pre_path /path/to/activation_tensors_mean.pt \
	--model_save_path /path/to/sae_model.pth
```

Notes:
- Adjust `--nproc_per_node` to your GPU count.
- If you want to resume, pass `--model_load_path /path/to/checkpoint.pth`.

## sae.py

- Implements the `TopKSparseAutoencoder` model and its encode/decode logic.
- Normalizes inputs, applies Top-K sparsity, and reconstructs with a tied decoder.
- Includes utilities for decoder normalization and gradient projection.
- Provides `load_sae_model()` to load a saved checkpoint with configuration inferred from weights.
