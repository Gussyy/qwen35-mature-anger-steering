"""Phase D: render sweep JSONL as human-readable markdown for Claude-as-judge
to score, and load scored cells back for best-cell selection.

Score schema (written to judge_log.jsonl, one row per cell):
  {"layer": int, "coef": float, "mature_anger": float, "juvenile_rage": float,
   "coherence": float, "notes": str}
Scores are 1..5 means over the 10 generations in the cell.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

from config import DATA_DIR


def render_markdown(sweep_path: Path, out_md: Path, max_gen_chars: int = 400) -> None:
    rows = [json.loads(l) for l in sweep_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["layer"], r["coef"]))
    lines: List[str] = [f"# Sweep: {sweep_path.name}", ""]
    for r in rows:
        tag = f"L={r['layer']} c={r['coef']:+.1f}"
        lines.append(f"## {tag}  ppl={r['ppl']:.2f} ({r['ppl_ratio']:.2f}x)  collapsed={r['collapsed']}")
        if r["collapsed"]:
            lines.append("_(collapsed — no generations)_\n")
            continue
        for g in r["generations"]:
            text = (g["gen"] or "").strip().replace("\n", " ")
            if len(text) > max_gen_chars:
                text = text[:max_gen_chars] + "…"
            lines.append(f"- **{g['prompt']}** → {text}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[judge] wrote {out_md}")


def load_scores(path: Path) -> dict[tuple[int, float], dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out[(int(obj["layer"]), float(obj["coef"]))] = obj
    return out


def pick_best(sweep_path: Path, scores_path: Path, min_coherence: float = 3.5) -> tuple[int, float, dict] | None:
    scores = load_scores(scores_path)
    rows = [json.loads(l) for l in sweep_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    best = None
    for r in rows:
        if r["collapsed"]:
            continue
        if r["ppl_ratio"] > 1.5:
            continue
        key = (int(r["layer"]), float(r["coef"]))
        s = scores.get(key)
        if not s:
            continue
        if s["coherence"] < min_coherence:
            continue
        margin = s["mature_anger"] - s["juvenile_rage"]
        cand = {"layer": r["layer"], "coef": r["coef"], "margin": margin,
                "mature_anger": s["mature_anger"], "juvenile_rage": s["juvenile_rage"],
                "coherence": s["coherence"], "ppl_ratio": r["ppl_ratio"]}
        if best is None or cand["margin"] > best["margin"]:
            best = cand
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small"], default="large")
    ap.add_argument("--mode", choices=["render", "pick"], default="render")
    args = ap.parse_args()

    tag = "qwen_large" if args.model == "large" else "qwen_small"
    sweep = DATA_DIR / f"sweep_{tag}.jsonl"
    md = DATA_DIR / f"sweep_{tag}.md"
    scores = DATA_DIR / f"judge_{tag}.jsonl"

    if args.mode == "render":
        render_markdown(sweep, md)
    else:
        b = pick_best(sweep, scores)
        print(json.dumps(b, indent=2) if b else "no valid cell scored yet")


if __name__ == "__main__":
    main()
