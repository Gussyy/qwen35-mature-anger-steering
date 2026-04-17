"""Phase I.4: Multi-layer CAA stacking on 0.8B.

Apply the 0.8B's native CAA vectors at all 5 sweep layers simultaneously,
with coefficients scaled inversely by vector norm so each layer contributes
the same injected residual norm. Sweep overall alpha."""
from __future__ import annotations
import argparse, json
from contextlib import ExitStack
from pathlib import Path
import torch

from config import MODEL_SMALL, VECTORS_DIR, DATA_DIR, SWEEP_LAYERS
from model_loader import load, steering_hook
from perplexity import perplexity


@torch.no_grad()
def gen_multi(lm, layer_vec_coef, prompt, max_new=80):
    try:
        text = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n"
    ids = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    with ExitStack() as stack:
        for L, v, c in layer_vec_coef:
            stack.enter_context(steering_hook(lm, L, v, c))
        out = lm.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=lm.tokenizer.pad_token_id)
    return lm.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def ppl_multi(lm, layer_vec_coef):
    from contextlib import ExitStack
    with ExitStack() as stack:
        for L, v, c in layer_vec_coef:
            stack.enter_context(steering_hook(lm, L, v, c))
        return perplexity(lm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--n-prompts", type=int, default=15)
    args = ap.parse_args()

    prompts = [json.loads(l)["scenario"]
               for l in (DATA_DIR / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]

    print("[multi] loading 0.8B", flush=True)
    lm = load(MODEL_SMALL)

    # Load all 5 small CAA vectors
    layer_vecs = []
    for L in SWEEP_LAYERS:
        d = torch.load(VECTORS_DIR / f"qwen_small_L{L}_caa.pt", weights_only=True)
        layer_vecs.append((L, d["vector"], d["norm"]))

    base_ppl = perplexity(lm)
    print(f"[multi] base PPL = {base_ppl:.2f}  layer norms = {[(L, f'{n:.2f}') for L,_,n in layer_vecs]}", flush=True)

    out_path = DATA_DIR / "multi_layer_eval.jsonl"
    if out_path.exists(): out_path.unlink()

    for alpha in args.alphas:
        # Coefficient per layer = alpha / norm_L (so each layer contributes `alpha` injected norm)
        layer_vec_coef = [(L, v, alpha / n) for (L, v, n) in layer_vecs]
        label = f"multi_layer_a{alpha:.2f}"
        ppl = ppl_multi(lm, layer_vec_coef)
        if ppl > 3 * base_ppl:
            print(f"  {label}: COLLAPSED (ppl={ppl:.2f}, {ppl/base_ppl:.2f}x)", flush=True)
            row = {"label": label, "alpha": alpha,
                   "per_layer_coefs": {L: alpha/n for (L,_,n) in layer_vecs},
                   "ppl": ppl, "ppl_ratio": ppl/base_ppl, "base_ppl": base_ppl,
                   "collapsed": True, "generations": []}
        else:
            gens = []
            for p in prompts:
                g = gen_multi(lm, layer_vec_coef, p)
                gens.append({"prompt": p, "gen": g})
            row = {"label": label, "alpha": alpha,
                   "per_layer_coefs": {L: alpha/n for (L,_,n) in layer_vecs},
                   "ppl": ppl, "ppl_ratio": ppl/base_ppl, "base_ppl": base_ppl,
                   "collapsed": False, "generations": gens}
            print(f"  {label}: ppl={ppl:.2f} ({ppl/base_ppl:.2f}x)  ngen={len(gens)}", flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[multi] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
