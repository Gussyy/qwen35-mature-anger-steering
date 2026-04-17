"""Phase C: Compute Contrastive Activation Addition (CAA) vectors.

For each (scenario, mature_mad, neutral) triple we build two full sequences
and capture the residual stream at a target layer, averaged over the completion
tokens only. Per-pair difference (mad - neutral) is then averaged across the
dataset to produce the steering vector per layer.

Saves one .pt per layer to vectors/{model_tag}_L{N}_caa.pt.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import List
import torch
from tqdm import tqdm

from config import DATA_DIR, VECTORS_DIR, MODEL_IDS, MODEL_TAGS, SWEEP_LAYER_SETS
from model_loader import load, capture_residual


def load_pairs(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def completion_resid_mean(lm, layer_idx: int, prompt: str, completion: str) -> torch.Tensor:
    """Tokenize prompt+completion, forward, return mean residual over the
    completion-token positions at layer_idx. Shape: (d_model,)."""
    try:
        try:
            full_text = lm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completion}],
                tokenize=False, enable_thinking=False,
            )
            prompt_text = lm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            full_text = lm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completion}],
                tokenize=False,
            )
            prompt_text = lm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
    except Exception:
        full_text = prompt + "\n" + completion
        prompt_text = prompt + "\n"
    prompt_ids = lm.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = lm.tokenizer(full_text, return_tensors="pt")["input_ids"]
    p_len = prompt_ids.shape[1]
    # Completion starts at position p_len (inclusive)
    full_ids = full_ids.to(lm.model.device)
    collected: list[torch.Tensor] = []
    with capture_residual(lm, layer_idx, collected), torch.no_grad():
        lm.model(full_ids)
    resid = collected[0]  # (1, seq, d_model)
    comp = resid[0, p_len:]
    if comp.shape[0] == 0:
        comp = resid[0, -1:]
    return comp.mean(dim=0).float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small", "xlarge"], default="large")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    model_tag = MODEL_TAGS[args.model]
    layers = args.layers or SWEEP_LAYER_SETS[args.model]

    print(f"[caa] loading {model_id}")
    lm = load(model_id)
    print(f"[caa] d_model={lm.d_model} n_layers={lm.n_layers}")

    pairs = load_pairs(DATA_DIR / "contrast_pairs.jsonl")
    print(f"[caa] {len(pairs)} contrast pairs")

    for L in layers:
        t0 = time.time()
        deltas: list[torch.Tensor] = []
        for pair in tqdm(pairs, desc=f"layer {L}"):
            mad = completion_resid_mean(lm, L, pair["scenario"], pair["mature_mad"])
            neu = completion_resid_mean(lm, L, pair["scenario"], pair["neutral"])
            deltas.append((mad - neu).float())
        v = torch.stack(deltas).mean(dim=0)
        out_path = VECTORS_DIR / f"{model_tag}_L{L}_caa.pt"
        torch.save({"vector": v, "norm": float(v.norm()), "layer": L, "model": model_id, "n_pairs": len(pairs)}, out_path)
        print(f"[caa] L={L} saved norm={v.norm():.3f} in {time.time()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
