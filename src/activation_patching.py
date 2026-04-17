"""Phase I.2: Activation patching — ceiling test.

Idea: skip the linear-map-on-mean-difference. For each eval prompt:
  1. Run 2B (with native L=14 CAA at c=+1) to generate a completion; capture
     per-token residual stream at L=14 for the full generated sequence.
  2. Run 0.8B on the SAME prompt; at each forward step during generation,
     replace (patch) the 0.8B's L=14 residual with a projection of the 2B's
     captured residual at the corresponding token.

Because d_model differs (2048 vs 1024), we need a projection. Try three:
  A.1 ridge W (already fit)
  A.2 Procrustes R
  A.3 PCA — project 2B residual onto top-1024 SVD directions of an activation
       matrix; use those as a "natural" 1024-dim subspace of 2B

For fidelity we keep it simple: offline pass 1 — run 2B steered, save
residual sequences per prompt; offline pass 2 — run 0.8B for each prompt
with a hook that patches in the projected residual token-by-token.

Evaluated on 15 held-out prompts.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from config import MODEL_LARGE, MODEL_SMALL, VECTORS_DIR, DATA_DIR
from model_loader import load, capture_residual, steering_hook
from perplexity import perplexity


@torch.no_grad()
def capture_2b_steered(lm_2b, vec_2b, layer, c, prompts, max_new=80):
    """Run 2B with steering, capture per-token residuals at `layer` for
    prompt+generation. Returns list of dicts {prompt, token_ids, residual (seq, 2048)}.
    """
    out = []
    for p in prompts:
        try:
            text = lm_2b.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception:
            text = p + "\n"
        ids = lm_2b.tokenizer(text, return_tensors="pt").to(lm_2b.model.device)
        # Generate under steering
        with steering_hook(lm_2b, layer, vec_2b, c):
            gen = lm_2b.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=lm_2b.tokenizer.pad_token_id)
        # Forward full sequence under the same steering, capture residuals
        collected = []
        with steering_hook(lm_2b, layer, vec_2b, c), capture_residual(lm_2b, layer, collected):
            lm_2b.model(gen)
        resid = collected[0][0].detach().float().cpu()  # (seq, 2048)
        out.append({"prompt": p, "token_ids": gen[0].cpu().tolist(), "residual": resid})
    return out


def apply_patch_hook(lm, layer, patch_resid, scale=1.0):
    """Hook that REPLACES the layer's residual with patch_resid (same seq len)."""
    target = patch_resid.to(device=lm.model.device,
                            dtype=next(lm.model.parameters()).dtype)

    def hook(_m, _inp, out):
        resid = out[0] if isinstance(out, tuple) else out
        seq = resid.shape[1]
        patch = target[:seq].unsqueeze(0)  # (1, seq, d)
        # Replace (additive form): resid + (patch - resid) * scale
        new_resid = resid + (patch - resid) * scale
        if isinstance(out, tuple):
            return (new_resid,) + out[1:]
        return new_resid
    h = lm.layers[layer].register_forward_hook(hook)
    return h


@torch.no_grad()
def generate_patched(lm_08b, patched_resid_08b, prompt, layer, max_new=80):
    """Run the 0.8B with a hook that replaces residuals at `layer` with the
    precomputed patched_resid_08b (already in 1024-dim). For the prompt-only
    portion we replace; for newly-generated tokens we fall back to the normal
    residual (since we don't have a projection for future tokens).

    Simpler: use static_resid for first N tokens where N = len(patched), then
    let the model continue from there.
    """
    try:
        text = lm_08b.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n"
    ids = lm_08b.tokenizer(text, return_tensors="pt").to(lm_08b.model.device)
    p_len = ids["input_ids"].shape[1]

    target = patched_resid_08b.to(device=lm_08b.model.device,
                                  dtype=next(lm_08b.model.parameters()).dtype)

    def hook(_m, _inp, out):
        resid = out[0] if isinstance(out, tuple) else out
        seq = resid.shape[1]
        # Only patch prompt positions (first p_len tokens) — let the model
        # generate fresh after that, so the patched prompt biases the state.
        patch_len = min(seq, target.shape[0], p_len)
        if patch_len > 0:
            patch = target[:patch_len].unsqueeze(0)
            resid = resid.clone()
            resid[:, :patch_len, :] = patch
        if isinstance(out, tuple):
            return (resid,) + out[1:]
        return resid

    h = lm_08b.layers[layer].register_forward_hook(hook)
    try:
        gen = lm_08b.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=lm_08b.tokenizer.pad_token_id)
    finally:
        h.remove()
    gen_ids = gen[0, ids["input_ids"].shape[1]:]
    return lm_08b.tokenizer.decode(gen_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--c-2b", type=float, default=1.0)
    ap.add_argument("--n-prompts", type=int, default=15)
    args = ap.parse_args()

    prompts_path = DATA_DIR / "eval_prompts.jsonl"
    prompts = [json.loads(l)["scenario"]
               for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]

    # -- Pass 1: 2B steered, capture residuals --
    print("[patch] loading 2B", flush=True)
    lm_2b = load(MODEL_LARGE)
    v_2b = torch.load(VECTORS_DIR / f"qwen_large_L{args.layer}_caa.pt",
                      weights_only=True)["vector"]
    t0 = time.time()
    caps = capture_2b_steered(lm_2b, v_2b, args.layer, args.c_2b, prompts)
    del lm_2b; torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"[patch] captured 2B residuals for {len(caps)} prompts in {time.time()-t0:.1f}s", flush=True)

    # -- Projections --
    tm = torch.load(VECTORS_DIR / "transfer_map_large_to_small.pt", weights_only=False)
    W = torch.from_numpy(tm["W"]).float()   # (2048, 1024)
    R = torch.from_numpy(tm["R"]).float()   # (1024, 2048)

    # -- Pass 2: 0.8B with patch --
    print("[patch] loading 0.8B", flush=True)
    lm_08 = load(MODEL_SMALL)
    base_ppl = perplexity(lm_08)
    print(f"[patch] 0.8B base PPL = {base_ppl:.2f}", flush=True)

    out_path = DATA_DIR / "patch_eval.jsonl"
    if out_path.exists(): out_path.unlink()

    for label, proj in [("patch_ridge", W), ("patch_procrustes_fwd", R)]:
        print(f"[patch] variant {label}", flush=True)
        gens = []
        for c in caps:
            r2b = c["residual"]  # (seq, 2048)
            if label == "patch_ridge":
                r08 = (r2b @ proj).float()  # (seq, 1024)
            else:
                r08 = (r2b @ proj.T).float()  # (seq, 1024)
            gen = generate_patched(lm_08, r08, c["prompt"], args.layer)
            gens.append({"prompt": c["prompt"], "gen": gen})
        row = {"label": label, "layer": args.layer, "c_2b": args.c_2b,
               "generations": gens, "base_ppl_08": base_ppl}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {label}: {len(gens)} generations", flush=True)

    print(f"[patch] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
