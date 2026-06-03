import argparse
import json
import logging
import statistics
from pathlib import Path

import torch

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from question_datasets.openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from question_datasets import build_combined_question_dataset
from utils.llama_3_model_download import MODEL_REGISTRY, ensure_model_downloaded

from sae import load_sae_model
from utils.grounding_scores import GroundingScoreCalculator
from grounding_functions.curv import PseudoCurvConfig, compute_pseudo_curv
from grounding_functions.perplexity_regression import load_perplexity_regression_weights
from grounding_functions.stability import StabilityConfig, compute_fisher_diag, score_stability_delta


def parse_activation_indices(raw: str) -> list[int]:
    if not raw:
        return []
    if raw.strip().lower() in {"all", "*"}:
        return []
    return [int(idx.strip()) for idx in raw.split(",") if idx.strip()]


def find_activation_indices(activation_dir: Path, layer: int) -> list[int]:
    layer_dir = activation_dir / f"layer_{layer}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Activation layer directory not found: {layer_dir}")

    indices: list[int] = []
    prefix = f"activations_l{layer}_idx"
    for path in layer_dir.glob(f"{prefix}*.pt"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        raw_idx = stem[len(prefix) :]
        try:
            indices.append(int(raw_idx))
        except ValueError:
            continue

    return sorted(indices)


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding functions on a single activation")
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY.keys()), default=None)
    parser.add_argument("--sae_model_path", type=Path, required=True)
    parser.add_argument("--activation_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--metadata_file", type=Path, default=Path("activation_outs/metadata_rank0.jsonl"))
    parser.add_argument(
        "--activation_indices",
        type=str,
        default="3",
        help="Comma-separated indices, or 'all' to scan activation_dir/layer_{layer}.",
    )
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--avg_latents_dir", type=Path, default=Path("sparse_activation_analysis"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("results/grounding_eval.jsonl"))
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_batch_size", type=int, default=1)
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
    parser.add_argument("--stab_topk", type=int, default=20)
    parser.add_argument("--stab_alpha", type=float, default=0.5)
    parser.add_argument("--curv_topk_vocab", type=int, default=50)
    parser.add_argument("--curv_mc_samples", type=int, default=4)
    parser.add_argument("--curv_beta", type=float, default=0.01)
    parser.add_argument("--curv_latent_topk", type=int, default=8)
    parser.add_argument("--weight_sae", type=float, default=1.0)
    parser.add_argument("--weight_stab", type=float, default=1.0)
    parser.add_argument("--weight_curv", type=float, default=1.0)
    parser.add_argument(
        "--perplexity_weights_path",
        type=Path,
        default=None,
        help="Optional regression weights to score SAE latents against perplexity.",
    )
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
    args.output_jsonl = args.output_jsonl.resolve()
    if args.perplexity_weights_path is not None:
        args.perplexity_weights_path = args.perplexity_weights_path.resolve()

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

    regression_weights = None
    if args.perplexity_weights_path is not None:
        logging.info("Loading regression weights from %s", args.perplexity_weights_path)
        regression_weights = load_perplexity_regression_weights(
            args.perplexity_weights_path,
            device=device,
            dtype=torch.float32,
        )

    grounding_calc = GroundingScoreCalculator.from_avg_latents(
        avg_latents_dir=args.avg_latents_dir,
        device=device,
        dtype=sae_dtype,
    )

    activation_indices = parse_activation_indices(args.activation_indices)
    if not activation_indices:
        logging.info("No explicit indices provided; scanning activation directory...")
        activation_indices = find_activation_indices(args.activation_dir, args.layer)
    if not activation_indices:
        raise ValueError("No activation indices found")

    prompt_map: dict[int, str] = {}
    if args.use_metadata_prompt:
        logging.info("Loading prompts from metadata...")
        prompt_map = load_prompts_from_metadata(args.metadata_file, set(activation_indices))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    grounding_sae_values: list[float] = []
    grounding_stab_values: list[float] = []
    grounding_curv_values: list[float] = []
    grounding_total_values: list[float] = []
    grounding_reg_values: list[float] = []

    for sample_idx in activation_indices:
        activation_path = resolve_activation_path(args.activation_dir, args.layer, sample_idx)
        if not activation_path.exists():
            logging.warning("Activation file not found for idx %s: %s", sample_idx, activation_path)
            continue

        activations = load_activation_tensor(activation_path, device=device, dtype=model_dtype)
        if activations.dim() != 2:
            raise ValueError(f"Expected activations shape (seq_len, d_model), got {tuple(activations.shape)}")

        seq_len, _ = activations.shape
        token_pos = choose_token_pos(args.token_pos, seq_len)

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
            token_pos = choose_token_pos(token_pos, seq_len)

        override_base = activations.unsqueeze(0)
        h_token = override_base[:, token_pos].to(device=device, dtype=sae_dtype)

        with torch.no_grad():
            _, h_dense, h_sparse = sae.forward_1d_normalized(h_token)
            grounding_sae = grounding_calc.score(h_sparse).item()

        grounding_reg = None
        if regression_weights is not None:
            grounding_reg = regression_weights.predict(h_sparse).item()

        stability_cfg = StabilityConfig(topk=args.stab_topk)
        fisher_diag = compute_fisher_diag(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=args.layer,
            override_activations=override_base,
            token_pos=token_pos,
            config=stability_cfg,
        )

        top_latent = torch.topk(h_dense.abs().squeeze(0), k=1).indices.item()
        direction = sae.decoder.weight[:, top_latent]
        direction = direction / (direction.norm() + 1e-12)
        delta = args.stab_alpha * direction
        grounding_stab = score_stability_delta(delta, fisher_diag).item()

        curv_cfg = PseudoCurvConfig(
            topk_vocab=args.curv_topk_vocab,
            mc_samples=args.curv_mc_samples,
            beta=args.curv_beta,
            latent_topk=args.curv_latent_topk,
        )
        grounding_curv = compute_pseudo_curv(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=args.layer,
            base_override_activations=override_base,
            token_pos=token_pos,
            h_dense=h_dense,
            decoder_weight=sae.decoder.weight,
            config=curv_cfg,
        ).item()

        composite = (
            args.weight_sae * grounding_sae
            + args.weight_stab * grounding_stab
            + args.weight_curv * grounding_curv
        )

        grounding_sae_values.append(grounding_sae)
        grounding_stab_values.append(grounding_stab)
        grounding_curv_values.append(grounding_curv)
        grounding_total_values.append(composite)
        if grounding_reg is not None:
            grounding_reg_values.append(grounding_reg)

        record = {
            "sample_idx": sample_idx,
            "activation_path": str(activation_path),
            "token_pos": token_pos,
            "grounding_sae": grounding_sae,
            "grounding_stab": grounding_stab,
            "grounding_curv": grounding_curv,
            "grounding_composite": composite,
            "stab_alpha": args.stab_alpha,
            "stab_topk": args.stab_topk,
            "curv_topk_vocab": args.curv_topk_vocab,
            "curv_mc_samples": args.curv_mc_samples,
            "curv_beta": args.curv_beta,
            "curv_latent_topk": args.curv_latent_topk,
        }
        if grounding_reg is not None and regression_weights is not None:
            record["grounding_regression"] = grounding_reg
            record["regression_target"] = regression_weights.target_key

        with args.output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if grounding_reg is None:
            logging.info(
                "Idx %s: G_sae=%.4f, G_stab=%.4f, G_curv=%.4f, G_total=%.4f",
                sample_idx,
                grounding_sae,
                grounding_stab,
                grounding_curv,
                composite,
            )
        else:
            logging.info(
                "Idx %s: G_sae=%.4f, G_stab=%.4f, G_curv=%.4f, G_reg=%.4f, G_total=%.4f",
                sample_idx,
                grounding_sae,
                grounding_stab,
                grounding_curv,
                grounding_reg,
                composite,
            )

    if grounding_total_values:
        avg_total = statistics.mean(grounding_total_values)
        median_total = statistics.median(grounding_total_values)

        median_sae = statistics.median(grounding_sae_values)
        median_stab = statistics.median(grounding_stab_values)
        median_curv = statistics.median(grounding_curv_values)

        def _safe_inv(value: float) -> float:
            return 0.0 if value == 0 else 1.0 / abs(value)

        suggested_weights = {
            "weight_sae": _safe_inv(median_sae),
            "weight_stab": _safe_inv(median_stab),
            "weight_curv": _safe_inv(median_curv),
        }

        logging.info("G_total avg=%.6f, median=%.6f", avg_total, median_total)
        logging.info(
            "Suggested weights (inverse-median normalization): sae=%.6f, stab=%.6f, curv=%.6f",
            suggested_weights["weight_sae"],
            suggested_weights["weight_stab"],
            suggested_weights["weight_curv"],
        )

    if grounding_reg_values:
        avg_reg = statistics.mean(grounding_reg_values)
        median_reg = statistics.median(grounding_reg_values)
        logging.info("G_reg avg=%.6f, median=%.6f", avg_reg, median_reg)


if __name__ == "__main__":
    main()
