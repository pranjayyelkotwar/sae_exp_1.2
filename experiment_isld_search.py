import argparse
import json
import logging
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from evaluate_grounding_functions import (
    build_dataset,
    choose_token_pos,
    load_activation_tensor,
    load_model,
    load_prompts_from_metadata,
    resolve_activation_path,
)
from experiment_sae_direction_perturb import write_jsonl
from grounding_functions.curv import PseudoCurvConfig
from grounding_functions.perplexity_regression import load_perplexity_regression_weights
from grounding_functions.stability import StabilityConfig
from inference_activations import generate_with_activation_override
from llama_3.args import ModelArgs
from llama_3.tokenizer import Tokenizer
from sae import load_sae_model
from search.evaluator import GroundingEvaluator, GroundingWeights
from search.evolutionary_search import ISLDEvolutionarySearch, SearchConfig
from search.sampler import SparseMutationSampler, SparseMutationSamplerConfig
from search.state import SearchState
from search.stopping import StoppingConfig
from utils.grounding_scores import GroundingScoreCalculator
from utils.llama_3_model_download import MODEL_REGISTRY, ensure_model_downloaded


def parse_activation_indices(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(idx.strip()) for idx in raw.split(",") if idx.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISLD evolutionary sparse search")
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY.keys()), default=None)
    parser.add_argument("--sae_model_path", type=Path, required=True)
    parser.add_argument("--activation_dir", type=Path, default=Path("activation_outs"))
    parser.add_argument("--metadata_file", type=Path, default=Path("activation_outs/metadata_rank0.jsonl"))
    parser.add_argument("--activation_indices", type=str, default="3")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--avg_latents_dir", type=Path, default=Path("sparse_activation_analysis"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("results/isld_search.jsonl"))
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument(
        "--generate_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate text output before and after ISLD search.",
    )
    parser.add_argument("--output_temperature", type=float, default=0.0)
    parser.add_argument("--output_top_p", type=float, default=0.9)
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
    parser.add_argument("--active_topk", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--min_active", type=int, default=1)
    parser.add_argument("--max_active", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--min_improvement", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target_grounding", type=float, default=None)
    parser.add_argument("--stab_topk", type=int, default=20)
    parser.add_argument("--curv_topk_vocab", type=int, default=50)
    parser.add_argument("--curv_mc_samples", type=int, default=4)
    parser.add_argument("--curv_beta", type=float, default=0.01)
    parser.add_argument("--curv_latent_topk", type=int, default=8)
    parser.add_argument("--weight_sae", type=float, default=1.0)
    parser.add_argument("--weight_stab", type=float, default=1.0)
    parser.add_argument("--weight_curv", type=float, default=1.0)
    parser.add_argument("--weight_reg", type=float, default=0.0)
    parser.add_argument(
        "--perplexity_weights_path",
        type=Path,
        default=None,
        help="Optional regression weights for perplexity-grounding.",
    )
    parser.add_argument(
        "--use_regression_grounding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use regression grounding as the primary score (sets weight_sae=0, weight_reg=1).",
    )
    parser.add_argument("--seed", type=int, default=0)
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

    grounding_calc = GroundingScoreCalculator.from_avg_latents(
        avg_latents_dir=args.avg_latents_dir,
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

    if args.use_regression_grounding:
        if regression_weights is None:
            raise ValueError("--use_regression_grounding requires --perplexity_weights_path")
        args.weight_sae = 0.0
        if args.weight_reg == 0.0:
            args.weight_reg = 1.0

    evaluator = GroundingEvaluator(
        grounding_calc=grounding_calc,
        regression_weights=regression_weights,
        weights=GroundingWeights(
            sae=args.weight_sae,
            stability=args.weight_stab,
            curvature=args.weight_curv,
            regression=args.weight_reg,
        ),
        stability_config=StabilityConfig(topk=args.stab_topk),
        curv_config=PseudoCurvConfig(
            topk_vocab=args.curv_topk_vocab,
            mc_samples=args.curv_mc_samples,
            beta=args.curv_beta,
            latent_topk=args.curv_latent_topk,
        ),
    )

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    sampler = SparseMutationSampler(
        config=SparseMutationSamplerConfig(
            num_candidates=args.num_candidates,
            min_active=args.min_active,
            max_active=args.max_active,
            beta=args.beta,
        ),
        generator=generator,
    )

    search = ISLDEvolutionarySearch(
        model=model,
        sae=sae,
        evaluator=evaluator,
        sampler=sampler,
        config=SearchConfig(active_topk=args.active_topk),
        stopping_config=StoppingConfig(
            max_iters=args.max_iters,
            min_improvement=args.min_improvement,
            patience=args.patience,
            target_grounding=args.target_grounding,
        ),
    )

    activation_indices = parse_activation_indices(args.activation_indices)
    if not activation_indices:
        raise ValueError("No activation indices provided")

    prompt_map: dict[int, str] = {}
    if args.use_metadata_prompt:
        logging.info("Loading prompts from metadata...")
        prompt_map = load_prompts_from_metadata(args.metadata_file, set(activation_indices))

    index_iter = activation_indices
    if tqdm is not None:
        index_iter = tqdm(activation_indices, desc="ISLD samples")

    for sample_idx in index_iter:
        activation_path = resolve_activation_path(args.activation_dir, args.layer, sample_idx)
        if not activation_path.exists():
            logging.warning("Activation file not found for idx %s: %s", sample_idx, activation_path)
            continue

        activations = load_activation_tensor(activation_path, device=device, dtype=model_dtype)
        if activations.dim() != 2:
            raise ValueError(f"Expected activations shape (seq_len, d_model), got {tuple(activations.shape)}")

        seq_len, _d_model = activations.shape
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

        hidden_state = activations[token_pos].to(device=device, dtype=sae_dtype)
        override_activations = activations.unsqueeze(0)

        h_dense, h_sparse = sae.forward_1d_normalized(hidden_state.unsqueeze(0))[-2:]
        g_sae = evaluator.compute_sae(h_sparse)
        g_reg = evaluator.compute_regression(h_sparse)
        g_stab = evaluator.compute_stability(
            delta=torch.zeros_like(hidden_state),
            fisher_diag=evaluator.compute_fisher_diag(
                model=model,
                prompt_tokens=prompt_tokens,
                override_layer=args.layer,
                override_activations=override_activations,
                token_pos=token_pos,
            ),
        )
        g_curv = evaluator.compute_curvature(
            model=model,
            prompt_tokens=prompt_tokens,
            override_layer=args.layer,
            override_activations=override_activations,
            token_pos=token_pos,
            h_dense=h_dense,
            decoder_weight=sae.decoder.weight,
        )
        base_score = evaluator.total(g_sae, g_stab, g_curv, g_reg).item()

        state = SearchState(
            prompt_tokens=prompt_tokens,
            hidden_state=hidden_state,
            override_activations=override_activations,
            grounding_score=float(base_score),
            step=0,
            trajectory_id=f"idx_{sample_idx}",
            token_pos=token_pos,
            override_layer=args.layer,
            metadata={
                "sample_idx": sample_idx,
                "activation_path": str(activation_path),
                "prompt_text": prompt_text,
            },
        )

        output_original = None
        if args.generate_output:
            output_original = generate_with_activation_override(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                override_layer=args.layer,
                override_activations=override_activations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.output_temperature,
                top_p=args.output_top_p,
            )

        final_state, history = search.run(state)

        output_final = None
        if args.generate_output:
            output_final = generate_with_activation_override(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                override_layer=args.layer,
                override_activations=final_state.override_activations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.output_temperature,
                top_p=args.output_top_p,
            )
        rows = []
        for row in history.to_jsonl_rows():
            row.update(
                {
                    "sample_idx": sample_idx,
                    "activation_path": str(activation_path),
                    "trajectory_id": final_state.trajectory_id,
                    "token_pos": token_pos,
                    "layer": args.layer,
                }
            )
            if args.generate_output:
                if row.get("step") == 0 and output_original is not None:
                    row["output_original"] = output_original
                if row.get("step") == final_state.step and output_final is not None:
                    row["output_final"] = output_final
                    row["output_changed"] = output_original != output_final
            rows.append(row)
        write_jsonl(args.output_jsonl, rows)

    logging.info("Done. Results written to %s", args.output_jsonl)


if __name__ == "__main__":
    main()
