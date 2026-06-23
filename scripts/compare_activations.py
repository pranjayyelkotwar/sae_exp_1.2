"""Capture activations for paraphrase texts and compare against search results.

This script reuses model-loading logic from `capture_activations.py` to load
the Llama-3 Transformer, captures residual activations for a chosen layer,
and compares pooled activations against hidden states stored in JSONL files
under a results directory. Results are written as a JSONL with ranked matches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

# reuse helper from capture_activations
from capture_activations import load_model
from llama_3.tokenizer import Tokenizer
from llama_3.args import ModelArgs


def read_paraphrases(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # fallback: treat whole line as text
                records.append({"text": line})
    return records


def capture_text_activations(
    texts: List[str],
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    layer: int,
    device: torch.device,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Run texts through the model and save pooled layer activations.

    Returns a list of metadata dictionaries for each captured text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = out_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    meta_records = []
    model.to(device)
    model.eval()

    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, bos=True, eos=True)
        # truncate if needed
        max_len = getattr(model.params, "max_seq_len", None)
        if max_len is not None and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]

        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

        # forward pass (model uses inference_mode)
        _ = model(tokens, start_pos=0)

        activs = model.get_layer_residual_activs()
        if layer not in activs:
            raise KeyError(f"Layer {layer} not captured by model (store_layer_activ).")
        # activs[layer] -> tensor shape (bsz, seqlen, dim)
        activ = activs[layer].cpu()

        # pooled representation: mean over sequence tokens
        pooled = activ.mean(dim=1).squeeze(0).numpy()

        pooled_path = layer_dir / f"pooled_paraphrase_{i}.npy"
        full_path = layer_dir / f"activations_paraphrase_{i}.pt"
        np.save(pooled_path, pooled)
        torch.save(activ, full_path)

        meta = {
            "idx": i,
            "text": text,
            "pooled_path": str(pooled_path),
            "activations_path": str(full_path),
        }
        meta_records.append(meta)

    # write metadata jsonl
    meta_path = out_dir / "metadata_paraphrases.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta_records:
            f.write(json.dumps(m) + "\n")

    return meta_records


def load_search_vectors(results_dir: Path) -> List[Dict[str, Any]]:
    """Load hidden_state vectors from all JSONL files in results_dir.

    Returns list of dicts with keys: 'file', 'line_idx', 'vector', and original record.
    """
    entries = []
    for p in sorted(results_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for li, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # heuristics: find a key that looks like hidden_state
                vec = None
                if "hidden_state" in rec:
                    vec = rec["hidden_state"]
                else:
                    # try to find the first list-of-floats valued key
                    for k, v in rec.items():
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                            vec = v
                            break
                if vec is None:
                    continue
                entries.append({"file": str(p), "line_idx": li, "vector": np.array(vec, dtype=np.float32), "record": rec})
    return entries


def rank_matches(pooled_vec: np.ndarray, candidates: List[Dict[str, Any]], top_k: int = 10):
    # compute L2 distances
    dists = []
    for c in candidates:
        v = c["vector"]
        # if shapes mismatch, try to reshape or skip
        if v.shape != pooled_vec.shape:
            # try to handle if candidate is longer: mean-pool it
            if v.ndim == 2:
                cv = v.mean(axis=0)
            else:
                # reshape mismatch, skip
                continue
        else:
            cv = v
        dist = float(np.linalg.norm(pooled_vec - cv))
        dists.append((dist, c))
    dists.sort(key=lambda x: x[0])
    top = [ {"distance": d, "file": c["file"], "line_idx": c["line_idx"], "grounding_score": c["record"].get("grounding_score")} for d, c in dists[:top_k] ]
    return top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--paraphrases", type=Path, default=Path("reverse_gen_exp/paraphrases.jsonl"))
    parser.add_argument("--results_dir", type=Path, default=Path("evolutionary_search_with_better_gf/results5"))
    parser.add_argument("--out_dir", type=Path, default=Path("activation_outs/paraphrases"))
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"

    tokenizer = Tokenizer(str(tokenizer_path))

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    model_args = ModelArgs(**params)

    device = torch.device(args.device)

    # load model (reusing capture_activations.load_model)
    model = load_model(model_path=model_path, model_args=model_args, store_layer_activ=[args.layer], device=device, dtype=torch.bfloat16 if torch.cuda.is_available() and device.type.startswith("cuda") else torch.float32)

    # read paraphrases
    paraphrase_records = read_paraphrases(args.paraphrases)
    texts = [r.get("text", "") for r in paraphrase_records]

    meta = capture_text_activations(texts, tokenizer, model, args.layer, device, args.out_dir)

    # load search vectors
    candidates = load_search_vectors(args.results_dir)

    # compute and save rankings
    out_rank_path = args.out_dir / "paraphrase_matches.jsonl"
    with out_rank_path.open("w", encoding="utf-8") as f:
        for m in meta:
            pooled = np.load(m["pooled_path"])
            matches = rank_matches(pooled, candidates, top_k=args.top_k)
            out = {"idx": m["idx"], "text": m["text"], "matches": matches}
            f.write(json.dumps(out) + "\n")

    print(f"Done. Results written to {out_rank_path}")


if __name__ == "__main__":
    main()
