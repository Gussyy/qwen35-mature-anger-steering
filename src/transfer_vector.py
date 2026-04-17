"""Phase G: fit a linear transfer map from 2B residual stream (2048-d) at L*
to 0.8B residual stream (1024-d) at L*_small, and evaluate the transferred
steering vector against the small model's native vector and a random baseline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import torch
from sklearn.linear_model import Ridge

from config import DATA_DIR, VECTORS_DIR, MODEL_LARGE, MODEL_SMALL
from model_loader import load, capture_residual, steering_hook
from steer_generate import steered_generate
from perplexity import perplexity


# Neutral prompts (topic-disjoint from anger) for fitting the alignment map
NEUTRAL_PROMPTS = [
    "Preheat the oven to 350 degrees and prepare the baking sheet.",
    "The stock market closed slightly higher today after a volatile session.",
    "Photosynthesis converts sunlight and carbon dioxide into glucose.",
    "The conference room is available from 2 to 3 on Thursday afternoon.",
    "A square has four equal sides and four right angles.",
    "Tokyo is the capital of Japan and one of the world's largest cities.",
    "Please submit your timesheet by 5 PM on Friday.",
    "The novel was published in 1953 and won several literary awards.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The new library opens at nine on weekdays and ten on Saturdays.",
    "Mount Everest stands at approximately 8,849 meters above sea level.",
    "The recipe calls for two cups of flour and one teaspoon of salt.",
    "Photons are elementary particles that transmit electromagnetic force.",
    "The committee meets on the second Tuesday of each month.",
    "The Pacific Ocean is the largest of the world's five oceans.",
    "Please cite three sources to support your argument.",
    "Gravity on Earth is approximately 9.8 meters per second squared.",
    "The museum is hosting an exhibition on medieval manuscripts this fall.",
    "The algorithm runs in O(n log n) time on average.",
    "The president delivered a short address on infrastructure spending.",
    "Every triangle has interior angles that sum to 180 degrees.",
    "The restaurant specializes in regional cuisine and fresh seafood.",
    "The bookstore discounts hardcovers by 20 percent on weekends.",
    "The train leaves from platform seven at a quarter past four.",
    "A decimal point separates the whole-number part from the fractional part.",
    "The new policy takes effect at the start of the fiscal year.",
    "Photographs from the expedition will be published next month.",
    "The annual festival draws visitors from across the region.",
    "Mitochondria generate energy through oxidative phosphorylation.",
    "The quarterly report is available on the company website.",
]


def _mean_prompt_resid(lm, layer_idx: int, text: str) -> torch.Tensor:
    """Return the mean residual-stream vector over the prompt tokens at layer_idx."""
    try:
        prompt_text = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = text + "\n"
    ids = lm.tokenizer(prompt_text, return_tensors="pt").to(lm.model.device)
    collected: list[torch.Tensor] = []
    with capture_residual(lm, layer_idx, collected), torch.no_grad():
        lm.model(**ids)
    resid = collected[0][0].float().cpu()  # (seq, d_model)
    return resid.mean(dim=0)


def _collect_matrix(model_id: str, layer_idx: int, prompts: List[str]) -> np.ndarray:
    lm = load(model_id)
    rows = [_mean_prompt_resid(lm, layer_idx, p).numpy() for p in prompts]
    del lm
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return np.stack(rows)


def fit_transfer_map(L_large: int, L_small: int, alpha: float = 1.0) -> dict:
    print(f"[transfer] collecting 2B residuals at L={L_large}", flush=True)
    X = _collect_matrix(MODEL_LARGE, L_large, NEUTRAL_PROMPTS)
    print(f"[transfer] collecting 0.8B residuals at L={L_small}", flush=True)
    Y = _collect_matrix(MODEL_SMALL, L_small, NEUTRAL_PROMPTS)
    # Ridge: Y = X @ W  =>  W shape (2048, 1024)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, Y)
    W = ridge.coef_.T.astype(np.float32)  # (in_dim, out_dim)
    b = ridge.intercept_.astype(np.float32)
    # Also compute Procrustes (orthogonal) via truncated SVD of Y^T X
    M = Y.T @ X  # (1024, 2048)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = (U @ Vt).astype(np.float32)  # (1024, 2048); we want X @ R.T -> (n, 1024)
    return {"W": W, "b": b, "R": R, "X": X, "Y": Y,
            "L_large": L_large, "L_small": L_small}


def apply_and_eval(L_small: int, c_small: float, vec_small_native: torch.Tensor,
                   v_transferred: torch.Tensor, label: str,
                   prompts: List[str]) -> dict:
    """Apply a steering vector to the 0.8B model at (L_small, c_small)
    after rescaling it to match ||vec_small_native||. Returns PPL + generations."""
    lm = load(MODEL_SMALL)
    base_ppl = perplexity(lm)
    scale = vec_small_native.norm().item() / max(v_transferred.norm().item(), 1e-8)
    v = v_transferred * scale
    with steering_hook(lm, L_small, v, c_small):
        ppl = perplexity(lm)
    gens = []
    for p in prompts:
        gen = steered_generate(lm, v, L_small, c_small, p, max_new_tokens=80, do_sample=False)
        gens.append({"prompt": p, "gen": gen})
    del lm; torch.cuda.empty_cache()
    import gc; gc.collect()
    return {"label": label, "ppl": ppl, "ppl_ratio": ppl/base_ppl,
            "norm": float(v.norm()), "generations": gens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-large", type=int, required=True)
    ap.add_argument("--l-small", type=int, required=True)
    ap.add_argument("--c-small", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-eval", type=int, default=15)
    args = ap.parse_args()

    # Load vectors
    v_large = torch.load(VECTORS_DIR / f"qwen_large_L{args.l_large}_caa.pt", weights_only=True)["vector"]
    v_small = torch.load(VECTORS_DIR / f"qwen_small_L{args.l_small}_caa.pt", weights_only=True)["vector"]

    t = fit_transfer_map(args.l_large, args.l_small, alpha=args.alpha)
    W = torch.from_numpy(t["W"])           # (2048, 1024)
    b = torch.from_numpy(t["b"])           # (1024,)
    R = torch.from_numpy(t["R"])           # (1024, 2048)

    # Ridge-projected vector (ignore intercept — we only want direction)
    v_ridge = (v_large.float() @ W)              # (1024,)
    v_proc = (R @ v_large.float())                # (1024,)
    v_rand = torch.randn_like(v_small) * (v_small.norm() / (v_small.numel() ** 0.5))

    map_path = VECTORS_DIR / "transfer_map_large_to_small.pt"
    torch.save({"W": t["W"], "b": t["b"], "R": t["R"],
                "L_large": args.l_large, "L_small": args.l_small,
                "alpha": args.alpha}, map_path)
    print(f"[transfer] saved {map_path}", flush=True)
    tr_path = VECTORS_DIR / "qwen_small_transferred_caa.pt"
    torch.save({"v_ridge": v_ridge, "v_procrustes": v_proc, "v_random": v_rand,
                "v_small_native": v_small, "v_large_source": v_large,
                "L_small": args.l_small, "c_small": args.c_small}, tr_path)

    # Evaluate all four on 0.8B
    prompts = [json.loads(l)["scenario"] for l in (DATA_DIR / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_eval]
    results = {}
    for label, v in [("native", v_small.float()), ("ridge", v_ridge),
                     ("procrustes", v_proc), ("random", v_rand)]:
        print(f"[transfer] evaluating {label}", flush=True)
        results[label] = apply_and_eval(args.l_small, args.c_small, v_small.float(), v, label, prompts)

    out = DATA_DIR / "transfer_eval.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for label, r in results.items():
            f.write(json.dumps({"label": label, **r}, ensure_ascii=False) + "\n")
    print(f"[transfer] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
