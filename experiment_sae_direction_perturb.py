import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from question_datasets.openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from question_datasets import build_combined_question_dataset
from utils.llama_3_model_download import MODEL_REGISTRY, ensure_model_downloaded

from sae import load_sae_model
from inference_activations import generate_with_activation_override


def g_sae(z: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (c * z).sum(dim=-1) / (z.sum(dim=-1) + eps)


@dataclass
class PerturbationResult:
    sample_idx: int
    activation_path: str
    token_pos: int
    latent_idx: int
    sign: int
    alpha: float
    grounding_original: float
    grounding_new: float
    output_original: str
    output_new: str
    output_changed: bool


@dataclass
class TokenPerturbationSummary:
    sample_idx: int
    activation_path: str
    token_pos: int
    seq_len: int
    prompt_len: int
    top_latents: list[int]
    grounding_original: float
    avg_grounding_new: float
    avg_grounding_delta: float
    max_grounding_delta: float
    min_grounding_delta: float
    total_trials: int
    output_changed_count: int
    output_changed_ratio: float


def parse_activation_indices(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(idx.strip()) for idx in raw.split(",") if idx.strip()]


def load_model(
    model_path: Path,
    model_args: ModelArgs,
    device: torch.device,
    max_batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Transformer:
    logging.info("Initializing model on CPU...")
    torch.set_default_dtype(dtype)
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model = Transformer(model_args)

    logging.info("Loading model weights into CPU memory...")
    model_weights = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    logging.info("Loading model weights into model...")
    model.load_state_dict(model_weights)
    del model_weights

    logging.info("Moving model to device %s...", device)
    model.to(device)
    model.eval()

    logging.info("Model created successfully.")
    return model


def build_dataset(args: argparse.Namespace, tokenizer: Tokenizer):
    if args.dataset_source == "qa":
        dataset_names = [name.strip() for name in args.qa_datasets.split(",") if name.strip()]
        qa_num_samples = None
        if args.qa_num_samples:
            qa_num_samples = {}
            for spec in args.qa_num_samples.split(","):
                name, num = spec.strip().split(":")
                qa_num_samples[name.strip()] = int(num)
        return build_combined_question_dataset(
            dataset_names=dataset_names,
            tokenizer=tokenizer,
            max_token_length=args.max_token_length,
            add_bos_token=args.add_bos_token,
            include_choices=args.include_choices,
            num_samples=qa_num_samples,
        )

    return OpenWebTextSentencesDataset(
        tokenizer=tokenizer,
        max_token_length=args.max_token_length,
        num_samples=args.num_samples,
        shuffle=False,
        add_bos_token=args.add_bos_token,
    )


def load_prompts_from_metadata(
    metadata_file: Path,
    target_indices: set[int],
) -> dict[int, str]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    prompts: dict[int, str] = {}
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            capture_idx = record.get("capture_dataset_idx")
            prompt_text = record.get("prompt_text")
            if capture_idx is None or prompt_text is None:
                continue

            if int(capture_idx) in target_indices:
                prompts[int(capture_idx)] = prompt_text
                if len(prompts) == len(target_indices):
                    break

    return prompts


def resolve_activation_path(activation_dir: Path, layer: int, sample_idx: int) -> Path:
    return activation_dir / f"layer_{layer}" / f"activations_l{layer}_idx{sample_idx}.pt"


def load_activation_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    activ = torch.load(path, map_location="cpu", weights_only=True)
    return activ.to(device=device, dtype=dtype)


def choose_token_pos(token_pos: int, seq_len: int) -> int:
    if token_pos < 0:
        return max(0, seq_len - 1)
    return min(token_pos, max(0, seq_len - 1))


def encode_with_sae(
    sae,
    h_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h_dense, h_sparse = sae.forward_1d_normalized(h_vec)
    return h_dense, h_sparse


def build_decoder_direction(sae, latent_idx: int, eps: float = 1e-12) -> torch.Tensor:
    direction = sae.decoder.weight[:, latent_idx]
    norm = direction.norm() + eps
    return direction / norm


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturb SAE directions and observe grounding and outputs")
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY.keys()), default=None)
    parser.add_argument("--sae_model_path", type=Path, required=True)
    parser.add_argument("--activation_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--metadata_file", type=Path, default=Path("activation_outs/metadata_rank0.jsonl"))
    parser.add_argument("--activation_indices", type=str, default="3,2896,5312")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--avg_latents_dir", type=Path, default=Path("sparse_activation_analysis"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/sae_direction_perturb"))
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dataset_source", type=str, choices=["openwebtext", "qa"], default="qa")
    parser.add_argument(
        "--qa_datasets",
        type=str,
        default="arc_easy,mmlu,hle",
        help="Comma-separated list chosen from: arc_easy,mmlu,hle",
    )
    parser.add_argument(
        "--qa_num_samples",
        type=str,
        default=None,
        help="Limit samples per QA dataset. Format: 'dataset_name:num,dataset_name:num'",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument(
        "--use_metadata_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use prompt_text from metadata_rank0.jsonl instead of rebuilding datasets.",
    )
    parser.add_argument(
        "--add_bos_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add BOS token when encoding prompts.",
    )
    parser.add_argument(
        "--include_choices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include choices in QA prompts when available.",
    )
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sae_dtype", type=str, default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()
    if args.model_dir is None and args.model_name is None:
        raise ValueError("Either --model_dir or --model_name must be provided")
    if args.model_dir is not None and args.model_name is not None:
        raise ValueError("Provide only one of --model_dir or --model_name")

    if args.model_name is not None:
        args.model_dir = ensure_model_downloaded(args.model_name)
    else:
        args.model_dir = args.model_dir.resolve()

    args.activation_dir = args.activation_dir.resolve()
    args.metadata_file = args.metadata_file.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.output_jsonl is None:
        args.output_jsonl = args.output_dir / "results.jsonl"
    else:
        args.output_jsonl = args.output_jsonl.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.model_dtype]
    sae_dtype = torch.float32 if args.sae_dtype == "float32" else torch.float16

    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer(str(args.model_dir / "tokenizer.model"))

    dataset = None
    if not args.use_metadata_prompt:
        logging.info("Building dataset...")
        dataset = build_dataset(args, tokenizer)

    logging.info("Loading model params and weights...")
    with (args.model_dir / "params.json").open("r", encoding="utf-8") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)

    target_seq_len = args.max_token_length + args.max_new_tokens
    model_max_seq_len = max(model_args.max_seq_len, target_seq_len)

    model = load_model(
        model_path=args.model_dir / "consolidated.00.pth",
        model_args=model_args,
        device=device,
        max_batch_size=args.max_batch_size,
        max_seq_len=model_max_seq_len,
        dtype=model_dtype,
    )

    logging.info("Loading SAE model...")
    sae = load_sae_model(
        model_path=args.sae_model_path,
        sae_top_k=8,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=sae_dtype,
    )

    arc_path = args.avg_latents_dir / "avg_latents_arc_easy.pt"
    hle_path = args.avg_latents_dir / "avg_latents_hle.pt"

    if not arc_path.exists() or not hle_path.exists():
        raise FileNotFoundError("Missing avg latents; expected ARC-Easy and HLE tensors in avg_latents_dir")

    arc = torch.load(arc_path, map_location="cpu", weights_only=True).to(device=device, dtype=sae_dtype)
    hle = torch.load(hle_path, map_location="cpu", weights_only=True).to(device=device, dtype=sae_dtype)

    if arc.shape != hle.shape:
        raise ValueError(f"Shape mismatch: ARC-Easy {tuple(arc.shape)} vs HLE {tuple(hle.shape)}")

    c_vector = arc - hle

    activation_indices = parse_activation_indices(args.activation_indices)
    if not activation_indices:
        raise ValueError("No activation indices provided")

    prompt_map: dict[int, str] = {}
    if args.use_metadata_prompt:
        logging.info("Loading prompts from metadata...")
        prompt_map = load_prompts_from_metadata(args.metadata_file, set(activation_indices))

    alphas = [0.1, 0.2, 0.5, 1.0, 2.0]
    signs = [1, -1]

    meta_path = args.output_dir / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "activation_indices": activation_indices,
                "activation_dir": str(args.activation_dir),
                "layer": args.layer,
                "token_pos": args.token_pos,
                "model_dir": str(args.model_dir),
                "sae_model_path": str(args.sae_model_path),
                "avg_latents_dir": str(args.avg_latents_dir),
                "dataset_source": args.dataset_source,
                "qa_datasets": args.qa_datasets,
                "qa_num_samples": args.qa_num_samples,
                "max_token_length": args.max_token_length,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "model_dtype": args.model_dtype,
                "sae_dtype": args.sae_dtype,
            },
            f,
            indent=2,
        )

    for sample_idx in activation_indices:
        activation_path = resolve_activation_path(args.activation_dir, args.layer, sample_idx)
        if not activation_path.exists():
            logging.warning("Activation file not found for idx %s: %s", sample_idx, activation_path)
            continue

        activations = load_activation_tensor(activation_path, device=device, dtype=model_dtype)
        if activations.dim() != 2:
            raise ValueError(f"Expected activations shape (seq_len, d_model), got {tuple(activations.shape)}")

        seq_len, d_model = activations.shape
        prompt_text = None
        if args.use_metadata_prompt:
            prompt_text = prompt_map.get(sample_idx)
            if prompt_text is None:
                logging.warning("Missing prompt_text for idx %s in metadata; falling back to dataset", sample_idx)

        if prompt_text is not None:
            prompt_tokens = tokenizer.encode(prompt_text, bos=args.add_bos_token, eos=False)
            prompt_tokens = prompt_tokens[: args.max_token_length]
        else:
            if dataset is None:
                raise ValueError("Dataset not initialized; disable --use_metadata_prompt to rebuild datasets")
            prompt_tokens, _idx, prompt_len, _metadata = dataset[sample_idx]
            if len(prompt_tokens) != prompt_len:
                prompt_tokens = prompt_tokens[:prompt_len]

        if len(prompt_tokens) != seq_len:
            min_len = min(len(prompt_tokens), seq_len)
            logging.warning(
                "Length mismatch for idx %s (tokens=%s, activations=%s); truncating to %s",
                sample_idx,
                len(prompt_tokens),
                seq_len,
                min_len,
            )
            prompt_tokens = prompt_tokens[:min_len]
            activations = activations[:min_len]
            seq_len = min_len

        override_base = activations.unsqueeze(0)

        with torch.no_grad():
            output_original = generate_with_activation_override(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                override_layer=args.layer,
                override_activations=override_base,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        rows = []
        for token_pos in range(seq_len):
            h_token = activations[token_pos].to(device=device, dtype=sae_dtype).unsqueeze(0)

            with torch.no_grad():
                h_dense, h_sparse = encode_with_sae(sae, h_token)
                grounding_original = g_sae(h_sparse, c_vector).item()

                _, top_indices = torch.topk(h_dense.abs().squeeze(0), k=5)
                top_indices = top_indices.tolist()

                total_trials = 0
                output_changed_count = 0
                grounding_new_sum = 0.0
                grounding_delta_sum = 0.0
                max_grounding_delta = float("-inf")
                min_grounding_delta = float("inf")

                for latent_idx in top_indices:
                    direction = build_decoder_direction(sae, latent_idx)

                    for sign in signs:
                        for alpha in alphas:
                            h_new = h_token.squeeze(0) + (sign * alpha * direction)

                            with torch.no_grad():
                                _, h_sparse_new = encode_with_sae(sae, h_new.unsqueeze(0))
                                grounding_new = g_sae(h_sparse_new, c_vector).item()

                            override_activations = activations.clone()
                            override_activations[token_pos] = h_new.to(device=device, dtype=model_dtype)
                            override_activations = override_activations.unsqueeze(0)

                            with torch.no_grad():
                                output_new = generate_with_activation_override(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt_tokens=prompt_tokens,
                                    override_layer=args.layer,
                                    override_activations=override_activations,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                )

                            total_trials += 1
                            grounding_new_sum += grounding_new
                            delta = grounding_new - grounding_original
                            grounding_delta_sum += delta
                            max_grounding_delta = max(max_grounding_delta, delta)
                            min_grounding_delta = min(min_grounding_delta, delta)

                            if output_new != output_original:
                                output_changed_count += 1

                avg_grounding_new = grounding_new_sum / max(1, total_trials)
                avg_grounding_delta = grounding_delta_sum / max(1, total_trials)
                output_changed_ratio = output_changed_count / max(1, total_trials)

                summary = TokenPerturbationSummary(
                    sample_idx=sample_idx,
                    activation_path=str(activation_path),
                    token_pos=token_pos,
                    seq_len=seq_len,
                    prompt_len=len(prompt_tokens),
                    top_latents=top_indices,
                    grounding_original=float(grounding_original),
                    avg_grounding_new=float(avg_grounding_new),
                    avg_grounding_delta=float(avg_grounding_delta),
                    max_grounding_delta=float(max_grounding_delta),
                    min_grounding_delta=float(min_grounding_delta),
                    total_trials=total_trials,
                    output_changed_count=output_changed_count,
                    output_changed_ratio=float(output_changed_ratio),
                )
                rows.append(asdict(summary))

            write_jsonl(args.output_jsonl, rows)
            logging.info("Wrote %s token summaries for sample %s", len(rows), sample_idx)

    logging.info("Done. Results written to %s", args.output_jsonl)


if __name__ == "__main__":
    main()
