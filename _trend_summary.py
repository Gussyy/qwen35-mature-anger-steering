"""Summarize the trend across 0.8B / 2B / 4B — CAA norms, sweep PPLs, DeepSeek scores."""
import json, sys, io, torch
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=== CAA norms ladder ===")
layer_set = {"qwen_small": [6,10,14,18,22], "qwen_large": [6,10,14,18,22], "qwen_xlarge": [13,18,23]}
for tag, layers in layer_set.items():
    print(f"\n{tag}:")
    for L in layers:
        p = Path(f"vectors/{tag}_L{L}_caa.pt")
        if not p.exists(): continue
        d = torch.load(p, weights_only=True)
        print(f"  L={L:>2}  norm={d['norm']:.3f}  d_model={d['vector'].numel()}")

print("\n=== Sweep + DeepSeek judge ladder (positive-coef cells only) ===")
for tag in ["qwen_small", "qwen_large", "qwen_xlarge"]:
    sweep = Path(f"data/sweep_{tag}.jsonl")
    tagshort = tag.replace("qwen_","")
    judge_path = Path(f"data/judge_deepseek_{tagshort}.jsonl")
    if not sweep.exists(): continue
    sr = [json.loads(l) for l in sweep.read_text(encoding="utf-8").splitlines() if l.strip()]
    jr = {}
    if judge_path.exists():
        for line in judge_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                jr[(int(o["layer"]), float(o["coef"]))] = o
    sr.sort(key=lambda r: (r["layer"], r["coef"]))
    best_margin = None; best_cell = None
    print(f"\n{tag}  (base ppl: {sr[0]['base_ppl']:.2f})")
    for r in sr:
        if r["coef"] < 0: continue
        if r["collapsed"]: continue
        key = (int(r["layer"]), float(r["coef"]))
        j = jr.get(key, {})
        ma = j.get("mature_anger", None)
        jrage = j.get("juvenile_rage", None)
        co = j.get("coherence", None)
        margin = (ma - jrage) if (ma is not None and jrage is not None) else None
        print(f"  L={r['layer']:>2} c={r['coef']:+.1f}  ppl={r['ppl']:>6.2f} ({r['ppl_ratio']:.2f}x)  ma={ma} jr={jrage} co={co} margin={margin}")
        if margin is not None and co is not None and co >= 3.5:
            if best_margin is None or margin > best_margin:
                best_margin, best_cell = margin, (r["layer"], r["coef"], ma, jrage, co, r["ppl_ratio"])
    if best_cell:
        print(f"  BEST (coh>=3.5): L={best_cell[0]} c={best_cell[1]:+.1f}  ma={best_cell[2]} jr={best_cell[3]} co={best_cell[4]} margin=+{best_margin}  ppl={best_cell[5]:.2f}x")
