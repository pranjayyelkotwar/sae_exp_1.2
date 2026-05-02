# utils

Concise utilities for Llama 3 model download, local path handling, and CUDA inference setup.

## Core functionality
- Download minimal Llama 3 model artifacts from Hugging Face and cache locally.
- Provide wrappers to resolve local model directories for specific Llama 3 variants.
- Configure CUDA flags and seeds for inference reproducibility/perf.

## Files

| File | Purpose | Key entry points |
| --- | --- | --- |
| `__init__.py` | Package marker (empty). | N/A |
| `cuda_utils.py` | CUDA flags and seed setup for inference. | `set_up_cuda()`, `set_torch_seed_for_inference(seed)` |
| `llama_3_model_download.py` | Model registry, HF download, and CLI. | `MODEL_REGISTRY`, `ensure_model_downloaded()`, `main()` |
| `model_wrappers.py` | Thin wrappers for Llama 3 model directory resolution. | `Llama3Instruct8BWrapper`, `Llama31Instruct8BWrapper` |


## Impactful areas to extend
- Model coverage: add new entries to `MODEL_REGISTRY` and `LLAMA_MODEL_FILES` in `llama_3_model_download.py` to support other Llama variants or different artifact layouts.
- Storage layout: update `download_dir` per model or change `ensure_model_downloaded()` to customize cache location, naming, or mirroring strategy.
- Auth and distribution: extend `ensure_model_downloaded()` to accept an explicit token, alternate env vars, or offline mirrors; add retries/validation for file integrity.
- Wrapper behavior: extend `model_wrappers.py` to expose tokenizer paths, config paths, or to accept a pre-downloaded directory; add new wrapper classes for additional models.
- CUDA behavior: adjust `set_up_cuda()` flags (TF32, cuDNN benchmark, SDP) and `set_torch_seed_for_inference()` determinism settings for your perf vs. reproducibility tradeoffs.

## CLI usage

```bash
python -m utils.llama_3_model_download --model llama_3-8B
```

## Constraints
- Requires `HF_TOKEN` env var (or pass a token into `ensure_model_downloaded()` if you extend it).
- Expects Hugging Face repos to expose `original/` artifacts listed in `LLAMA_MODEL_FILES`.
- Only models in `MODEL_REGISTRY` are supported by default.
