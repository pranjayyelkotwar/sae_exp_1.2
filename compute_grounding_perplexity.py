import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from utils.cuda_utils import set_up_cuda
from utils.llama_3_model_download import MODEL_REGISTRY, ensure_model_downloaded


@dataclass
class MetadataRecord:
    line_idx: int
    capture_dataset_idx: int
    capture_layer: int | None
    prompt_text: str
    activation_path: str | None
    activation_path_resolved: str | None
    source_dataset: str | None
    source_id: str | None
    token_count: int | None
    compute_key: int


@dataclass
class TokenizedItem:
    key: int
    tokens: list[int]
    seq_len: int
    token_count_meta: int | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute prompt perplexity from metadata_rank0.jsonl")
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY.keys()), default=None)
    parser.add_argument("--metadata_file", type=Path, default=Path("activation_outs/metadata_rank0.jsonl"))
    parser.add_argument("--activation_out_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("results/grounding_perplexity.jsonl"))
    parser.add_argument("--layer", type=int, default=None, help="Only process records with this capture_layer.")
    parser.add_argument("--max_records", type=int, default=None, help="Limit number of metadata rows processed.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Override model max_batch_size (defaults to batch_size).",
    )
    parser.add_argument(
        "--seq_chunk_size",
        type=int,
        default=64,
        help="Chunk length for logits to reduce VRAM (set to max_token_length for full chunk).",
    )
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument(
        "--add_bos_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match activation capture tokenization (add BOS).",
    )
    parser.add_argument(
        "--include_prompt_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include prompt_text in output JSONL.",
    )
    parser.add_argument(
        "--no_dedupe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute perplexity per metadata row instead of per capture_dataset_idx.",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def resolve_activation_path(
    activation_out_dir: Path,
    activation_path: str | None,
    layer: int | None,
) -> Path | None:
    if not activation_path:
        return None

    path = Path(activation_path)
    if not path.is_absolute():
        path = activation_out_dir / path

    if path.is_absolute() and not path.exists():
        parts = list(path.parts)
        if "activation_outs" in parts:
            idx = parts.index("activation_outs")
            path = activation_out_dir.joinpath(*parts[idx + 1 :])

    if not path.exists() and layer is not None:
        path = activation_out_dir / f"layer_{layer}" / path.name

    return path


def load_model(
    model_path: Path,
    model_args: ModelArgs,
    device: torch.device,
    max_batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
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


def iter_metadata_records(
    metadata_file: Path,
    activation_out_dir: Path,
    layer_filter: int | None,
    no_dedupe: bool,
    max_records: int | None,
) -> list[MetadataRecord]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    records: list[MetadataRecord] = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_records is not None and len(records) >= max_records:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt_text = record.get("prompt_text")
            if not prompt_text:
                continue

            capture_layer = record.get("capture_layer")
            if layer_filter is not None and capture_layer != layer_filter:
                continue

            capture_idx = record.get("capture_dataset_idx")
            if capture_idx is None:
                capture_idx = record.get("combined_dataset_idx")
            if capture_idx is None:
                continue

            activation_path = record.get("activation_path")
            resolved_path = resolve_activation_path(
                activation_out_dir=activation_out_dir,
                activation_path=activation_path,
                layer=capture_layer,
            )

            compute_key = int(line_idx) if no_dedupe else int(capture_idx)
            records.append(
                MetadataRecord(
                    line_idx=int(line_idx),
                    capture_dataset_idx=int(capture_idx),
                    capture_layer=int(capture_layer) if capture_layer is not None else None,
                    prompt_text=prompt_text,
                    activation_path=activation_path,
                    activation_path_resolved=str(resolved_path) if resolved_path is not None else None,
                    source_dataset=record.get("source_dataset"),
                    source_id=record.get("source_id"),
                    token_count=record.get("token_count"),
                    compute_key=compute_key,
                )
            )

    return records


def tokenize_prompts(
    tokenizer: Tokenizer,
    records: list[MetadataRecord],
    add_bos_token: bool,
    max_token_length: int,
) -> list[TokenizedItem]:
    seen: set[int] = set()
    items: list[TokenizedItem] = []
    for record in records:
        if record.compute_key in seen:
            continue
        seen.add(record.compute_key)

        tokens = tokenizer.encode(record.prompt_text, bos=add_bos_token, eos=False)
        tokens = tokens[:max_token_length]
        seq_len = len(tokens)

        if record.token_count is not None and record.token_count != seq_len:
            logging.warning(
                "Token count mismatch for idx=%s (meta=%s, encoded=%s)",
                record.capture_dataset_idx,
                record.token_count,
                seq_len,
            )

        items.append(
            TokenizedItem(
                key=record.compute_key,
                tokens=tokens,
                seq_len=seq_len,
                token_count_meta=record.token_count,
            )
        )
    return items


def batch_iterable(items: list[TokenizedItem], batch_size: int) -> Iterable[list[TokenizedItem]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def compute_perplexity(
    model: Transformer,
    items: list[TokenizedItem],
    pad_id: int,
    device: torch.device,
    batch_size: int,
    seq_chunk_size: int,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}

    iterator = batch_iterable(items, batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Perplexity", unit="batch")

    for batch in iterator:
        max_len = max(item.seq_len for item in batch)
        input_ids = torch.full(
            (len(batch), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.tensor([item.seq_len for item in batch], device=device)

        for i, item in enumerate(batch):
            if item.seq_len > 0:
                input_ids[i, : item.seq_len] = torch.tensor(
                    item.tokens,
                    dtype=torch.long,
                    device=device,
                )

        with torch.inference_mode():
            sum_nll = torch.zeros(len(batch), device=device)
            denom = torch.zeros(len(batch), device=device)

            start_pos = 0
            while start_pos < max_len:
                end_pos = min(start_pos + seq_chunk_size, max_len)
                chunk_len = end_pos - start_pos
                if chunk_len <= 0:
                    break

                token_chunk = input_ids[:, start_pos:end_pos]
                logits = model(token_chunk, start_pos=start_pos)

                max_label_len = min(chunk_len, max_len - start_pos - 1)
                if max_label_len <= 0:
                    start_pos = end_pos
                    continue

                logits = logits[:, :max_label_len, :]
                labels = input_ids[:, start_pos + 1 : start_pos + 1 + max_label_len]

                logsumexp = torch.logsumexp(logits, dim=-1)
                target_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                token_nll = logsumexp - target_logits

                positions = torch.arange(max_label_len, device=device).unsqueeze(0)
                valid_mask = positions < (lengths - 1 - start_pos).unsqueeze(1)
                token_nll = token_nll * valid_mask

                sum_nll += token_nll.sum(dim=1)
                denom += valid_mask.sum(dim=1)

                start_pos = end_pos

            denom_safe = denom.clamp(min=1)
            mean_nll = sum_nll / denom_safe
            mean_nll = torch.where(
                denom > 0,
                mean_nll,
                torch.full_like(mean_nll, float("nan")),
            )
            perplexity = torch.exp(mean_nll)

        for item, s_nll, m_nll, ppl in zip(batch, sum_nll, mean_nll, perplexity, strict=True):
            results[item.key] = {
                "sum_nll": float(s_nll.item()),
                "mean_nll": float(m_nll.item()),
                "perplexity": float(ppl.item()),
                "seq_len": float(item.seq_len),
            }

    return results


def parse_dtype(raw: str) -> torch.dtype:
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float16":
        return torch.float16
    return torch.float32


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

    args.metadata_file = args.metadata_file.resolve()
    args.activation_out_dir = args.activation_out_dir.resolve()
    args.output_jsonl = args.output_jsonl.resolve()

    max_batch_size = args.max_batch_size or args.batch_size
    device = torch.device(args.device)
    dtype = parse_dtype(args.model_dtype)

    if device.type == "cuda":
        set_up_cuda()

    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"

    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer(str(tokenizer_path))

    logging.info("Loading model params from %s...", params_path)
    with params_path.open("r", encoding="utf-8") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)

    model = load_model(
        model_path=model_path,
        model_args=model_args,
        device=device,
        max_batch_size=max_batch_size,
        max_seq_len=args.max_token_length,
        dtype=dtype,
    )

    logging.info("Loading metadata from %s...", args.metadata_file)
    records = iter_metadata_records(
        metadata_file=args.metadata_file,
        activation_out_dir=args.activation_out_dir,
        layer_filter=args.layer,
        no_dedupe=args.no_dedupe,
        max_records=args.max_records,
    )

    if not records:
        logging.warning("No metadata records found.")
        return

    logging.info("Tokenizing %s prompts...", len(records))
    tokenized_items = tokenize_prompts(
        tokenizer=tokenizer,
        records=records,
        add_bos_token=args.add_bos_token,
        max_token_length=args.max_token_length,
    )

    logging.info(
        "Computing perplexity for %s unique prompts (batch_size=%s)...",
        len(tokenized_items),
        args.batch_size,
    )
    metrics = compute_perplexity(
        model=model,
        items=tokenized_items,
        pad_id=tokenizer.pad_id,
        device=device,
        batch_size=args.batch_size,
        seq_chunk_size=args.seq_chunk_size,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            metric = metrics.get(record.compute_key)
            if metric is None:
                continue

            out = {
                "capture_dataset_idx": record.capture_dataset_idx,
                "capture_layer": record.capture_layer,
                "activation_path": record.activation_path,
                "activation_path_resolved": record.activation_path_resolved,
                "source_dataset": record.source_dataset,
                "source_id": record.source_id,
                "token_count_metadata": record.token_count,
                "token_count_encoded": int(metric["seq_len"]),
                "sum_nll": metric["sum_nll"],
                "mean_nll": metric["mean_nll"],
                "perplexity": metric["perplexity"],
            }
            if args.include_prompt_text:
                out["prompt_text"] = record.prompt_text

            f.write(json.dumps(out) + "\n")

    logging.info("Wrote output to %s", args.output_jsonl)


if __name__ == "__main__":
    main()
