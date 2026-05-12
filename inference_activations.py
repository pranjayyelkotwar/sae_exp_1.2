import argparse
import json
import logging
import re
import random
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from question_datasets.openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from question_datasets import build_combined_question_dataset
from utils.llama_3_model_download import MODEL_REGISTRY, ensure_model_downloaded

ACTIVATION_FILENAME_RE = re.compile(r"activations_l(?P<layer>\d+)_idx(?P<idx>\d+)\.pt")


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY.keys()), default=None)
    parser.add_argument("--activation_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--output_jsonl", type=Path, default=Path("activation_inference_results.jsonl"))
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Randomly subsample N activation files for a quick test run.",
    )
    parser.add_argument(
        "--subsample_seed",
        type=int,
        default=42,
        help="Random seed for subsampling activation files.",
    )
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
        help="Limit samples per QA dataset. Format: 'dataset_name:num,dataset_name:num' (e.g., 'mmlu:2200')",
    )
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Override model max_batch_size to reduce KV cache memory.",
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
    return parser.parse_args()


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def generate_with_activation_override(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt_tokens: list[int],
    override_layer: int,
    override_activations: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = next(model.parameters()).device
    prompt_len = len(prompt_tokens)
    total_seq_len = prompt_len + max_new_tokens
    if total_seq_len > model.params.max_seq_len:
        raise ValueError(
            f"Total sequence length {total_seq_len} exceeds model max_seq_len {model.params.max_seq_len}"
        )

    tokens = torch.full(
        (1, total_seq_len),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    tokens[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    input_tokens_mask = tokens != tokenizer.pad_id

    with torch.no_grad():
        outputs = model.forward_with_activation_override(
            tokens[:, 0:prompt_len],
            start_pos=0,
            override_layer=override_layer,
            override_activations=override_activations,
        )
        logits = outputs[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tokens = sample_top_p(probs, top_p).squeeze(-1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        if prompt_len < total_seq_len:
            tokens[:, prompt_len] = next_tokens

        prev_pos = prompt_len
        for cur_pos in range(prompt_len + 1, total_seq_len):
            outputs = model(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            logits = outputs[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_tokens = sample_top_p(probs, top_p).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            next_tokens = torch.where(
                input_tokens_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_tokens,
            )
            tokens[:, cur_pos] = next_tokens
            prev_pos = cur_pos

    generated = tokens[0, prompt_len:total_seq_len]
    generated = generated[generated != tokenizer.pad_id].tolist()
    return tokenizer.decode(generated)


@torch.inference_mode()
def logits_with_activation_override(
    model: Transformer,
    prompt_tokens: list[int],
    override_layer: int,
    override_activations: torch.Tensor,
    token_pos: int,
) -> torch.Tensor:
    if override_activations.dim() != 3:
        raise ValueError(
            "override_activations must have shape (batch, seq_len, d_model); "
            f"got {tuple(override_activations.shape)}"
        )

    batch_size, seq_len, _ = override_activations.shape
    if seq_len != len(prompt_tokens):
        raise ValueError(
            "prompt_tokens length must match override_activations seq_len; "
            f"got tokens={len(prompt_tokens)} and seq_len={seq_len}"
        )

    if token_pos < 0:
        token_pos = max(0, seq_len - 1)
    else:
        token_pos = min(token_pos, max(0, seq_len - 1))

    device = next(model.parameters()).device
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(batch_size, 1)

    outputs = model.forward_with_activation_override(
        tokens,
        start_pos=0,
        override_layer=override_layer,
        override_activations=override_activations,
    )
    return outputs[:, token_pos, :]


def parse_activation_idx(path: Path, expected_layer: int) -> int | None:
    match = ACTIVATION_FILENAME_RE.match(path.name)
    if not match:
        return None
    layer = int(match.group("layer"))
    if layer != expected_layer:
        return None
    return int(match.group("idx"))


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


def load_activation_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    activ = torch.load(path, map_location="cpu", weights_only=True)
    if activ.dtype != dtype:
        activ = activ.to(dtype)
    return activ.to(device)


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
    args.output_jsonl = args.output_jsonl.resolve()

    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"

    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer(str(tokenizer_path))

    logging.info("Loading model parameters...")
    with params_path.open("r") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    max_seq_len = min(
        model_args.max_seq_len,
        args.max_token_length + args.max_new_tokens,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path=model_path,
        model_args=model_args,
        device=device,
        max_batch_size=args.max_batch_size,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
    )

    logging.info("Building dataset...")
    dataset = build_dataset(args, tokenizer)

    layer_dir = args.activation_dir / f"layer_{args.layer}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Activation layer directory not found: {layer_dir}")

    activation_paths = sorted(layer_dir.glob("*.pt"))
    activation_paths = [
        path for path in activation_paths if parse_activation_idx(path, args.layer) is not None
    ]

    if args.subsample is not None:
        if args.subsample <= 0:
            raise ValueError("--subsample must be a positive integer")
        if args.subsample < len(activation_paths):
            rng = random.Random(args.subsample_seed)
            activation_paths = rng.sample(activation_paths, args.subsample)
        else:
            logging.info(
                "Requested subsample size %s exceeds available activations (%s). Using all.",
                args.subsample,
                len(activation_paths),
            )

    if args.num_samples is not None:
        activation_paths = activation_paths[: args.num_samples]

    if not activation_paths:
        raise FileNotFoundError(f"No activation files found under {layer_dir}")

    logging.info("Writing JSONL to %s", args.output_jsonl)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("w", encoding="utf-8") as jsonl_file:

        iterator = activation_paths
        if tqdm is not None:
            iterator = tqdm(activation_paths, desc="Running activation inference")

        for path in iterator:
            capture_idx = parse_activation_idx(path, args.layer)
            if capture_idx is None:
                continue
            if capture_idx >= len(dataset):
                logging.warning(
                    "Capture idx %s is out of dataset range (len=%s). Skipping.",
                    capture_idx,
                    len(dataset),
                )
                continue

            if args.dataset_source == "qa":
                prompt_tokens, _idx, seq_len, metadata = dataset[capture_idx]
                prompt_text = metadata.get("prompt_text", "")
                dataset_source = metadata.get("source_dataset", "qa")
                source_id = metadata.get("source_id", str(capture_idx))
                combined_dataset_idx = metadata.get("combined_dataset_idx", capture_idx)
            else:
                prompt_tokens, _idx, seq_len = dataset[capture_idx]
                prompt_text = tokenizer.decode(prompt_tokens)
                dataset_source = "openwebtext"
                source_id = str(capture_idx)
                combined_dataset_idx = capture_idx

            activ = load_activation_tensor(
                path,
                device=device,
                dtype=next(model.parameters()).dtype,
            )

            if activ.ndim == 2:
                activ = activ.unsqueeze(0)

            if activ.shape[1] != seq_len:
                min_len = min(activ.shape[1], seq_len)
                logging.warning(
                    "Activation length mismatch for idx %s: activ=%s, seq_len=%s. Trimming to %s.",
                    capture_idx,
                    activ.shape[1],
                    seq_len,
                    min_len,
                )
                activ = activ[:, :min_len]
                prompt_tokens = prompt_tokens[:min_len]
                seq_len = min_len

            answer = generate_with_activation_override(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                override_layer=args.layer,
                override_activations=activ,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "dataset_source": dataset_source,
                "source_id": source_id,
                "combined_dataset_idx": combined_dataset_idx,
                "capture_dataset_idx": capture_idx,
                "prompt": prompt_text,
                "final_answer": answer,
            }
            jsonl_file.write(json.dumps(record) + "\n")

            logging.info("Processed capture idx %s", capture_idx)


if __name__ == "__main__":
    main()
