"""DeepSeek-judge the 4 transfer variants (native, ridge, procrustes, random)."""
from __future__ import annotations
import json, os, sys, urllib.request, urllib.error, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

from deepseek_judge import RUBRIC, deepseek_chat, format_cell

inp = Path("data/transfer_eval.jsonl")
out = Path("data/judge_deepseek_transfer.jsonl")
if out.exists(): out.unlink()

rows = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    gens = r["generations"]
    user = format_cell(gens)
    j = deepseek_chat(RUBRIC, user)
    ma, jr, co = float(j["mature_anger"]), float(j["juvenile_rage"]), float(j["coherence"])
    row = {"label": r["label"], "ppl_ratio": r["ppl_ratio"],
           "mature_anger": ma, "juvenile_rage": jr, "coherence": co,
           "margin": ma - jr, "notes": j.get("notes", "")}
    out.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {r['label']:12s}  ma={ma:.1f} jr={jr:.1f} co={co:.1f} margin={ma-jr:+.1f}")
