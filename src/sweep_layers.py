"""Phase C: sweep (layer, coefficient) grid, generate 10 completions per cell,
log perplexity for collapse detection, dump all generations to JSONL.
Resumable: skips (layer, coef) cells already present in the output file."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import List, Set, Tuple
import torch

from config import (DATA_DIR, VECTORS_DIR, MODEL_IDS, MODEL_TAGS,
                    SWEEP_LAYER_SETS, SWEEP_COEFS)
from model_loader import load, steering_hook
from steer_generate import steered_generate
from perplexity import perplexity


def load_eval_prompts(n: int = 10) -> List[str]:
    path = DATA_DIR / "eval_prompts.jsonl"
    prompts = [json.loads(l)["scenario"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return prompts[:n]


def existing_cells(path: Path) -> Set[Tuple[int, float]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            done.add((int(obj["layer"]), float(obj["coef"])))
        except Exception:
            continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small", "xlarge"], default="large")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--coefs", nargs="*", type=float, default=None)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--reset", action="store_true", help="Delete previous output and start over")
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    model_tag = MODEL_TAGS[args.model]
    layers = args.layers or SWEEP_LAYER_SETS[args.model]
    coefs = args.coefs or SWEEP_COEFS
    prompts = load_eval_prompts(args.n_prompts)

    out_path = DATA_DIR / f"sweep_{model_tag}.jsonl"
    if args.reset and out_path.exists():
        out_path.unlink()
    done = existing_cells(out_path)
    print(f"[sweep] already done: {len(done)} cells", flush=True)

    total_cells = len(layers) * len(coefs)
    todo = [(L, c) for L in layers for c in coefs if (L, float(c)) not in done]
    print(f"[sweep] todo: {len(todo)}/{total_cells} cells", flush=True)
    if not todo:
        print("[sweep] nothing to do", flush=True)
        return

    print(f"[sweep] loading {model_id}", flush=True)
    lm = load(model_id)

    base_ppl = perplexity(lm)
    print(f"[sweep] baseline PPL = {base_ppl:.2f}", flush=True)

    t_all = time.time()
    vec_cache: dict[int, torch.Tensor] = {}
    for (L, c) in todo:
        if L not in vec_cache:
            vec_path = VECTORS_DIR / f"{model_tag}_L{L}_caa.pt"
            if not vec_path.exists():
                print(f"[sweep] SKIP L={L}: {vec_path} missing", flush=True)
                continue
            vec_cache[L] = torch.load(vec_path, weights_only=True)["vector"]
        vec = vec_cache[L]
        t0 = time.time()
        if c == 0.0:
            ppl = base_ppl
        else:
            with steering_hook(lm, L, vec, c):
                ppl = perplexity(lm)
        ppl_ratio = ppl / base_ppl
        collapsed = ppl_ratio > 3.0
        generations = []
        for i, p in enumerate(prompts):
            if collapsed:
                generations.append({"prompt_idx": i, "prompt": p, "gen": None})
                continue
            gen = steered_generate(lm, vec, L, c, p, max_new_tokens=args.max_new,
                                   do_sample=False)
            generations.append({"prompt_idx": i, "prompt": p, "gen": gen})
        cell = {
            "layer": L, "coef": float(c),
            "base_ppl": base_ppl, "ppl": ppl, "ppl_ratio": ppl_ratio,
            "collapsed": collapsed,
            "generations": generations,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cell, ensure_ascii=False) + "\n")
        ngen = sum(g['gen'] is not None for g in generations)
        print(f"[sweep] L={L} c={c:+.1f} ppl={ppl:.2f} ({ppl_ratio:.2f}x) collapsed={collapsed} "
              f"ngen={ngen} in {time.time()-t0:.1f}s", flush=True)
    print(f"[sweep] done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
