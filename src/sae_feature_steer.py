"""Phase I.3: SAE feature-level steering.

Instead of adding a steering vector to the residual stream, directly amplify
specific SAE features that correspond (by the latent-diff analysis) to the
2B's dedicated mature-anger feature 4617.

Mechanism:
  - Forward hook on layer 14.
  - Capture residual, pass through SAE encoder.
  - For each target feature in `feat_ids`, boost its pre-activation so the
    Top-K selection guarantees it fires at magnitude alpha.
  - Decode and return `resid + (reconstruction - sae_roundtrip(resid))` —
    i.e. add the delta the SAE produces when those features are clamped up.

The feature IDs come from latent_diff_analysis.py output — the 0.8B SAE
features with highest cosine similarity to the ridge-projected 2B feature
4617 decoder vector. Top 3 from I.1: feat 79 (+0.261), feat 973 (+0.212),
feat 1206 (+0.191).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from config import MODEL_SMALL, SAES_DIR, DATA_DIR
from model_loader import load
from train_sae import TopKSAE
from perplexity import perplexity


def feature_steering_hook(lm, layer, sae, feat_ids, alpha):
    """Returns handle; adds a hook that clamps selected SAE features to alpha,
    decodes, adds the reconstruction delta to the residual."""
    dev = lm.model.device
    feat_ids_t = torch.tensor(feat_ids, dtype=torch.long, device=dev)
    a_val = float(alpha)

    def hook(_m, _inp, out):
        resid = out[0] if isinstance(out, tuple) else out
        orig_shape = resid.shape
        x = resid.reshape(-1, orig_shape[-1]).float()
        pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
        # Force target features high in the pre-activation so they survive TopK
        pre[:, feat_ids_t] = a_val * 5.0  # big enough to dominate topK
        vals, idx = pre.topk(sae.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, torch.relu(vals))
        # Also directly clamp the target features' final activations to alpha
        batch_idx = torch.arange(acts.shape[0], device=dev).unsqueeze(1).expand(-1, len(feat_ids))
        acts[batch_idx, feat_ids_t] = a_val
        x_hat = acts @ sae.W_dec + sae.b_dec
        # Original roundtrip (without clamping) for delta
        vals0, idx0 = ((x - sae.b_dec) @ sae.W_enc + sae.b_enc).topk(sae.k, dim=-1)
        acts0 = torch.zeros_like(pre)
        acts0.scatter_(-1, idx0, torch.relu(vals0))
        x_hat0 = acts0 @ sae.W_dec + sae.b_dec
        delta = x_hat - x_hat0  # (N, d_model)
        new_resid = (x + delta).to(resid.dtype).reshape(orig_shape)
        if isinstance(out, tuple):
            return (new_resid,) + out[1:]
        return new_resid

    return lm.layers[layer].register_forward_hook(hook)


@torch.no_grad()
def gen_with_feature_steer(lm, sae, layer, feat_ids, alpha, prompt, max_new=80):
    try:
        text = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n"
    ids = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    h = feature_steering_hook(lm, layer, sae, feat_ids, alpha)
    try:
        out = lm.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=lm.tokenizer.pad_token_id)
    finally:
        h.remove()
    return lm.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-ids", type=int, nargs="+", default=[79, 973, 1206])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--n-prompts", type=int, default=15)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    prompts = [json.loads(l)["scenario"]
               for l in (DATA_DIR / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]

    print("[sae_steer] loading 0.8B", flush=True)
    lm = load(MODEL_SMALL)
    ck = torch.load(SAES_DIR / f"qwen_small_L{args.layer}_sae.pt", weights_only=True)
    sae = TopKSAE(ck["d_model"], ck["d_sae"], ck["k"]).to(lm.model.device).float()
    sae.load_state_dict(ck["state_dict"])
    sae.eval()

    base_ppl = perplexity(lm)
    print(f"[sae_steer] base PPL = {base_ppl:.2f}", flush=True)

    out_path = DATA_DIR / "sae_steer_eval.jsonl"
    if out_path.exists(): out_path.unlink()
    for alpha in args.alphas:
        label = f"sae_feat_steer_a{alpha:.1f}"
        # Measure PPL under this feature steer
        h = feature_steering_hook(lm, args.layer, sae, args.feat_ids, alpha)
        try:
            ppl = perplexity(lm)
        finally:
            h.remove()
        gens = []
        for p in prompts:
            g = gen_with_feature_steer(lm, sae, args.layer, args.feat_ids, alpha, p)
            gens.append({"prompt": p, "gen": g})
        row = {"label": label, "feat_ids": args.feat_ids, "alpha": alpha,
               "layer": args.layer, "ppl": ppl, "ppl_ratio": ppl/base_ppl,
               "base_ppl": base_ppl, "generations": gens}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {label}: ppl={ppl:.2f} ({ppl/base_ppl:.2f}x)  ngen={len(gens)}", flush=True)

    print(f"[sae_steer] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
