"""Dump self vs DeepSeek judge scores side by side for both sweeps."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

def load(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(int(r["layer"]), float(r["coef"])): r for r in rows}

for tag, short in [("qwen_large", "large"), ("qwen_small", "small")]:
    self_ = load(f"data/judge_{tag}.jsonl")
    ds    = load(f"data/judge_deepseek_{short}.jsonl")
    print(f"\n=== {tag} ===")
    print(f"{'cell':>12}  {'self ma':>8} {'ds ma':>6}  {'self jr':>8} {'ds jr':>6}  {'self co':>8} {'ds co':>6}  {'self m':>7} {'ds m':>6}")
    best_self, best_ds = None, None
    for key in sorted(set(self_) | set(ds)):
        s = self_.get(key, {})
        d = ds.get(key, {})
        sm = s.get("mature_anger",0) - s.get("juvenile_rage",0)
        dm = d.get("mature_anger",0) - d.get("juvenile_rage",0)
        skip = d.get("collapsed", False)
        ok_self = (s.get("coherence",0) >= 3.5)
        ok_ds   = (d.get("coherence",0) >= 3.5 and not skip)
        if ok_self and (best_self is None or sm > best_self[1]):
            best_self = (key, sm, s)
        if ok_ds and (best_ds is None or dm > best_ds[1]):
            best_ds = (key, dm, d)
        print(f"  L={key[0]:>2} c={key[1]:+.1f}  "
              f"{s.get('mature_anger',0):>6.1f}  {d.get('mature_anger',0):>5.1f}  "
              f"{s.get('juvenile_rage',0):>6.1f}  {d.get('juvenile_rage',0):>5.1f}  "
              f"{s.get('coherence',0):>6.1f}  {d.get('coherence',0):>5.1f}  "
              f"{sm:+6.1f}  {dm:+5.1f}")
    print(f"\n  best by self (coh>=3.5): L={best_self[0][0]} c={best_self[0][1]:+.1f} margin={best_self[1]:+.1f}")
    print(f"  best by ds   (coh>=3.5): L={best_ds[0][0]} c={best_ds[0][1]:+.1f} margin={best_ds[1]:+.1f}")

print("\n=== transfer ===")
for line in Path("data/judge_deepseek_transfer.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        print(f"  {r['label']:12s}  ds ma={r['mature_anger']:.1f} jr={r['juvenile_rage']:.1f} co={r['coherence']:.1f} margin={r['margin']:+.1f}  notes={r['notes'][:100]}")
