"""Phase F: produce a side-by-side comparison of the native CAA vectors /
best cells / PPL cost / SAE feature sets between 2B and 0.8B."""
from __future__ import annotations
import json
from pathlib import Path
import torch

from config import DATA_DIR, VECTORS_DIR, SAES_DIR, SWEEP_LAYERS


def summarize(model_tag: str) -> dict:
    sweep_path = DATA_DIR / f"sweep_{model_tag}.jsonl"
    rows = [json.loads(l) for l in sweep_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_layer: dict[int, dict] = {}
    for r in rows:
        L = int(r["layer"])
        per_layer.setdefault(L, {})[float(r["coef"])] = {
            "ppl": r["ppl"], "ppl_ratio": r["ppl_ratio"], "collapsed": r["collapsed"],
        }
    vectors = {}
    for L in SWEEP_LAYERS:
        p = VECTORS_DIR / f"{model_tag}_L{L}_caa.pt"
        if p.exists():
            d = torch.load(p, weights_only=True)
            vectors[L] = {"norm": d["norm"], "shape": tuple(d["vector"].shape)}
    return {"per_layer": per_layer, "vectors": vectors}


def main() -> None:
    out = {"large": summarize("qwen_large"), "small": summarize("qwen_small")}
    p = DATA_DIR / "size_comparison.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[compare] wrote {p}")


if __name__ == "__main__":
    main()
