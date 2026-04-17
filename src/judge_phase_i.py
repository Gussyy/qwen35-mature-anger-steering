"""Judge all Phase I JSONLs with DeepSeek and produce a unified table."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from deepseek_judge import RUBRIC, deepseek_chat, format_cell

DATA = Path("data")
OUT = DATA / "judge_phase_i.jsonl"
if OUT.exists(): OUT.unlink()

# Each input file has a different schema around `generations`; standardize.
INPUT_FILES = [
    ("patch_ridge",           DATA / "patch_eval.jsonl",       {"filter": lambda r: r["label"] == "patch_ridge"}),
    ("patch_procrustes_fwd",  DATA / "patch_eval.jsonl",       {"filter": lambda r: r["label"] == "patch_procrustes_fwd"}),
    ("sae_feat_a1.0",         DATA / "sae_steer_eval.jsonl",   {"filter": lambda r: abs(r["alpha"] - 1.0) < 1e-6}),
    ("sae_feat_a2.0",         DATA / "sae_steer_eval.jsonl",   {"filter": lambda r: abs(r["alpha"] - 2.0) < 1e-6}),
    ("sae_feat_a4.0",         DATA / "sae_steer_eval.jsonl",   {"filter": lambda r: abs(r["alpha"] - 4.0) < 1e-6}),
    ("multi_layer_a0.5",      DATA / "multi_layer_eval.jsonl", {"filter": lambda r: abs(r["alpha"] - 0.5) < 1e-6}),
    ("multi_layer_a1.0",      DATA / "multi_layer_eval.jsonl", {"filter": lambda r: abs(r["alpha"] - 1.0) < 1e-6}),
    ("multi_layer_a1.5",      DATA / "multi_layer_eval.jsonl", {"filter": lambda r: abs(r["alpha"] - 1.5) < 1e-6}),
    ("multi_layer_a2.0",      DATA / "multi_layer_eval.jsonl", {"filter": lambda r: abs(r["alpha"] - 2.0) < 1e-6}),
    ("prompt_only",           DATA / "anchor_eval.jsonl",      {"filter": lambda r: r["label"] == "prompt_only"}),
    ("steer_only",            DATA / "anchor_eval.jsonl",      {"filter": lambda r: r["label"] == "steer_only"}),
    ("prompt_and_steer",      DATA / "anchor_eval.jsonl",      {"filter": lambda r: r["label"] == "prompt_and_steer"}),
    ("no_prompt_no_steer",    DATA / "anchor_eval.jsonl",      {"filter": lambda r: r["label"] == "no_prompt_no_steer"}),
]

for label, path, cfg in INPUT_FILES:
    if not path.exists():
        print(f"  [skip] {label}: {path} missing", flush=True); continue
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if cfg["filter"](r)]
    if not rows:
        print(f"  [skip] {label}: no matching row", flush=True); continue
    r = rows[0]
    gens = r.get("generations", [])
    if not gens or all(not g.get("gen") for g in gens):
        out_row = {"label": label, "source": str(path.name),
                   "mature_anger": 1.0, "juvenile_rage": 1.0, "coherence": 1.0,
                   "margin": 0.0, "ppl_ratio": r.get("ppl_ratio"),
                   "notes": "no generations (collapsed or empty)"}
        OUT.open("a", encoding="utf-8").write(json.dumps(out_row, ensure_ascii=False) + "\n")
        print(f"  {label}: empty/collapsed", flush=True); continue
    user = format_cell(gens)
    j = deepseek_chat(RUBRIC, user)
    ma, jr, co = float(j["mature_anger"]), float(j["juvenile_rage"]), float(j["coherence"])
    out_row = {"label": label, "source": str(path.name),
               "mature_anger": ma, "juvenile_rage": jr, "coherence": co,
               "margin": ma - jr, "ppl_ratio": r.get("ppl_ratio"),
               "notes": j.get("notes", "")}
    OUT.open("a", encoding="utf-8").write(json.dumps(out_row, ensure_ascii=False) + "\n")
    print(f"  {label}: ma={ma:.1f} jr={jr:.1f} co={co:.1f} margin={ma-jr:+.1f}  ppl={r.get('ppl_ratio')}  notes={j.get('notes','')[:80]}", flush=True)

print(f"wrote {OUT}", flush=True)
