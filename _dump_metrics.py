"""Collect all training/eval metrics in one pass for the report."""
import json, torch, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V = Path("vectors"); S = Path("saes"); D = Path("data")

print("=== CAA vector norms ===")
for tag in ["qwen_large", "qwen_small"]:
    for L in [6, 10, 14, 18, 22]:
        p = V / f"{tag}_L{L}_caa.pt"
        if p.exists():
            d = torch.load(p, weights_only=True)
            print(f"  {tag} L={L:>2}  norm={d['norm']:.3f}  n_pairs={d['n_pairs']}  dim={d['vector'].numel()}")

print("\n=== SAE training final metrics ===")
for tag in ["qwen_large", "qwen_small"]:
    p = S / f"{tag}_L14_sae.pt"
    if p.exists():
        d = torch.load(p, weights_only=True)
        print(f"  {tag} L=14  d_model={d['d_model']}  d_sae={d['d_sae']}  k={d['k']}  final_recon={d['final_recon']:.5f}  steps={d['steps']}")

print("\n=== Transfer map metrics ===")
p = V / "transfer_map_large_to_small.pt"
if p.exists():
    d = torch.load(p, weights_only=False)
    import numpy as np
    W = d["W"] if isinstance(d["W"], np.ndarray) else d["W"].numpy()
    R = d["R"] if isinstance(d["R"], np.ndarray) else d["R"].numpy()
    print(f"  ridge W shape={W.shape}  frob_norm={np.linalg.norm(W):.3f}  alpha={d.get('alpha')}")
    print(f"  Procrustes R shape={R.shape}  frob_norm={np.linalg.norm(R):.3f}")

print("\n=== Transferred vector norms ===")
p = V / "qwen_small_transferred_caa.pt"
if p.exists():
    d = torch.load(p, weights_only=False)
    for key in ["v_small_native", "v_ridge", "v_procrustes", "v_random", "v_large_source"]:
        v = d[key]
        if hasattr(v, "float"):
            nrm = v.float().norm().item()
        else:
            import numpy as np
            nrm = float(np.linalg.norm(v))
        print(f"  {key:20s} dim={(v.numel() if hasattr(v,'numel') else v.size):>4}  norm={nrm:.3f}")

print("\n=== Sweep summary tables ===")
for tag in ["qwen_large", "qwen_small"]:
    p = D / f"sweep_{tag}.jsonl"
    if not p.exists(): continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["layer"], r["coef"]))
    print(f"\n  {tag}:  base_ppl={rows[0]['base_ppl']:.2f}")
    for r in rows:
        print(f"    L={r['layer']:>2} c={r['coef']:+.1f}  ppl={r['ppl']:>7.2f} ({r['ppl_ratio']:.2f}x)  collapsed={int(r['collapsed'])}")

print("\n=== Judge score summary ===")
for tag in ["qwen_large", "qwen_small"]:
    p = D / f"judge_{tag}.jsonl"
    if not p.exists(): continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n  {tag}:")
    for r in rows:
        margin = r["mature_anger"] - r["juvenile_rage"]
        print(f"    L={r['layer']:>2} c={r['coef']:+.1f}  ma={r['mature_anger']:.1f}  jr={r['juvenile_rage']:.1f}  coh={r['coherence']:.1f}  margin={margin:+.1f}")

print("\n=== Transfer judge ===")
p = D / "judge_transfer.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    print(f"  {r['label']:12s}  ppl={r['ppl_ratio']:.2f}x  ma={r['mature_anger']:.1f}  jr={r['juvenile_rage']:.1f}  coh={r['coherence']:.1f}  margin={r['margin']:+.1f}")
