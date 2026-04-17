"""Phase E: run eval prompts through the model with and without steering,
collect SAE feature activations at the target layer, rank top features by
(steered - unsteered) mean activation delta."""
from __future__ import annotations
import argparse, json
from contextlib import nullcontext
from pathlib import Path
import torch

from config import DATA_DIR, VECTORS_DIR, SAES_DIR, MODEL_LARGE, MODEL_SMALL
from model_loader import load, capture_residual, steering_hook
from train_sae import TopKSAE


def load_sae(path: Path, device: str):
    ck = torch.load(path, weights_only=True)
    sae = TopKSAE(ck["d_model"], ck["d_sae"], ck["k"]).to(device).float()
    sae.load_state_dict(ck["state_dict"])
    sae.eval()
    return sae, ck


@torch.no_grad()
def collect_features(lm, layer_idx: int, prompt: str, sae: TopKSAE,
                     steering_vec=None, coef: float = 0.0) -> torch.Tensor:
    """Return mean SAE feature activation over the full generated sequence."""
    try:
        text = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n"
    ids = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)

    # Step 1: generate 60 tokens (with or without steering hook)
    use_steer = steering_vec is not None and coef != 0.0
    if use_steer:
        with steering_hook(lm, layer_idx, steering_vec, coef):
            out = lm.model.generate(**ids, max_new_tokens=60, do_sample=False,
                                    pad_token_id=lm.tokenizer.pad_token_id)
    else:
        out = lm.model.generate(**ids, max_new_tokens=60, do_sample=False,
                                pad_token_id=lm.tokenizer.pad_token_id)

    # Step 2: forward the full generated sequence with the same steering in
    # place and capture residuals at layer_idx
    collected: list[torch.Tensor] = []
    if use_steer:
        with steering_hook(lm, layer_idx, steering_vec, coef), \
             capture_residual(lm, layer_idx, collected):
            lm.model(out)
    else:
        with capture_residual(lm, layer_idx, collected):
            lm.model(out)

    resid = collected[0][0]  # (seq, d_model)
    p_len = ids["input_ids"].shape[1]
    comp = resid[p_len:].float()
    if comp.shape[0] == 0:
        return torch.zeros(sae.d_sae)
    _, a, _ = sae(comp)
    return a.mean(dim=0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coef", type=float, required=True)
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=30)
    args = ap.parse_args()

    model_id = MODEL_LARGE if args.model == "large" else MODEL_SMALL
    model_tag = "qwen_large" if args.model == "large" else "qwen_small"
    lm = load(model_id)

    sae, sae_ck = load_sae(SAES_DIR / f"{model_tag}_L{args.layer}_sae.pt", lm.model.device)
    vec = torch.load(VECTORS_DIR / f"{model_tag}_L{args.layer}_caa.pt", weights_only=True)["vector"]

    prompts_path = DATA_DIR / "eval_prompts.jsonl"
    prompts = [json.loads(l)["scenario"] for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]

    steered_sum = torch.zeros(sae_ck["d_sae"])
    base_sum = torch.zeros(sae_ck["d_sae"])
    steered_count = torch.zeros(sae_ck["d_sae"])
    base_count = torch.zeros(sae_ck["d_sae"])
    for i, p in enumerate(prompts):
        s = collect_features(lm, args.layer, p, sae, steering_vec=vec, coef=args.coef)
        b = collect_features(lm, args.layer, p, sae)
        steered_sum += s
        base_sum += b
        steered_count += (s > 0).float()
        base_count += (b > 0).float()
        if (i + 1) % 5 == 0:
            print(f"[sae_ana] {i+1}/{len(prompts)}", flush=True)

    steered_mean = steered_sum / len(prompts)
    base_mean = base_sum / len(prompts)
    delta = steered_mean - base_mean
    steered_freq = steered_count / len(prompts)
    base_freq = base_count / len(prompts)

    topk = torch.topk(delta.abs(), args.top_k)
    ranked = []
    for idx in topk.indices.tolist():
        ranked.append({
            "feature_id": int(idx),
            "delta": float(delta[idx]),
            "steered_mean": float(steered_mean[idx]),
            "base_mean": float(base_mean[idx]),
            "steered_freq": float(steered_freq[idx]),
            "base_freq": float(base_freq[idx]),
        })
    out_path = SAES_DIR / f"{model_tag}_L{args.layer}_features.json"
    out_path.write_text(json.dumps({
        "model": model_id, "layer": args.layer, "coef": args.coef,
        "n_prompts": len(prompts), "top_features": ranked,
    }, indent=2), encoding="utf-8")
    print(f"[sae_ana] wrote {out_path}", flush=True)
    for r in ranked[:10]:
        print(f"  feat {r['feature_id']:>5}  delta={r['delta']:+.4f}  steered={r['steered_mean']:.4f}  base={r['base_mean']:.4f}  sfreq={r['steered_freq']:.2f}  bfreq={r['base_freq']:.2f}")


if __name__ == "__main__":
    main()
