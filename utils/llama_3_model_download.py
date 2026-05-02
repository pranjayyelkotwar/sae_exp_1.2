import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

LLAMA_MODEL_FILES = (
    "original/consolidated.00.pth",
    "original/params.json",
    "original/tokenizer.model",
)

MODEL_REGISTRY = {
    "llama_3-8B": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "download_dir": Path("llama_3-8B_model/"),
    },
    "llama_3.1-8B": {
        "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "download_dir": Path("llama_3.1-8B_model/"),
    },
    "llama_3.2-3B": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "download_dir": Path("llama_3.2-3B_model/"),
    },
}


def download_hf_hub_file(model_id: str, filename: str, token: str, download_dir: Path) -> None:
    """"""
    file_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=token,
        local_dir=download_dir,
    )
    print(f"{filename} downloaded to: {file_path}")


def get_model_config(model_name: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name]


def ensure_model_downloaded(model_name: str, token: str | None = None) -> Path:
    config = get_model_config(model_name)
    repo_id = config["repo_id"]
    download_dir = config["download_dir"].resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set HF_TOKEN environment variable")

    for filename in LLAMA_MODEL_FILES:
        target_path = download_dir / filename
        if target_path.exists():
            continue
        download_hf_hub_file(repo_id, filename, token, download_dir)

    return download_dir / "original"


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=sorted(MODEL_REGISTRY.keys()))
    return parser.parse_args()


def main() -> None:
    """"""
    args = parse_arguments()

    model_dir = ensure_model_downloaded(args.model)
    print(f"Model ready at: {model_dir}")


if __name__ == "__main__":
    main()
