"""Independent judging of sweep cells via DeepSeek API.

Reads a sweep JSONL, sends each cell's 10 generations to DeepSeek-chat
with a structured rubric, parses JSON scores, writes to a separate
judge file. Resumable: skips cells that have already been scored.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import List
import urllib.request
import urllib.error

from config import DATA_DIR
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY")
if not DEEPSEEK_KEY:
    sys.exit("DEEPSEEK_KEY missing from env")

RUBRIC = """You are an independent, careful evaluator for an LLM activation-steering experiment.

The target persona is "MATURE ANGER" — the tone of a seasoned adult (think 45-55, experienced professional) whose patience has finally run out. Markers: specific, calm-but-firm, measured, invokes time/history ("I've dealt with this for years"), cites concrete grievances, draws a hard line, does NOT shout, does NOT insult.

The failure mode to watch for is "JUVENILE RAGE" — cap-lock shouting, cruel name-calling ("ungrateful", "unlovable", "pathetic"), repeated outbursts, incoherent emotional flailing, personal attacks on the user themselves.

You will see N model outputs from ONE (layer, coefficient) cell of a steering sweep. All N were produced with identical steering settings on the SAME model, differing only in prompt. Score the cell AS A WHOLE on three axes, integers or halves 1.0-5.0:

- mature_anger: how strongly does the cell exhibit mature, measured, patience-out anger? (1 = none, 5 = every sample clearly shows it)
- juvenile_rage: how strongly does the cell tip into juvenile rage / cruelty / repetition / name-calling? (1 = none, 5 = dominant pattern)
- coherence: are outputs on-topic, grammatical, non-repetitive, stay within language norms? (1 = broken/looping, 5 = fully coherent)

Return STRICT JSON only, no prose, no markdown fence:
{"mature_anger": X, "juvenile_rage": X, "coherence": X, "notes": "one-sentence rationale citing a specific sample"}
"""


def deepseek_chat(system: str, user: str, model: str = "deepseek-chat",
                  max_retries: int = 5) -> dict:
    req_body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        DEEPSEEK_URL, data=req_body,
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                 "Content-Type": "application/json"},
        method="POST")
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [deepseek] retry {attempt+1}/{max_retries} after {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"deepseek api failed after {max_retries} retries: {last_err}")


def format_cell(gens: list) -> str:
    lines = []
    for i, g in enumerate(gens, 1):
        if g.get("gen") is None:
            continue
        text = g["gen"].replace("\n", " ").strip()
        if len(text) > 450:
            text = text[:450] + "..."
        lines.append(f"[{i}] scenario: {g['prompt']}")
        lines.append(f"    output:   {text}")
    return "\n".join(lines)


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            o = json.loads(line)
            done.add((int(o["layer"]), float(o["coef"])))
        except Exception:
            continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="path to sweep_*.jsonl")
    ap.add_argument("--out", required=True, help="path to judge_deepseek_*.jsonl")
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    out_path = Path(args.out)
    rows = [json.loads(l) for l in sweep_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = load_done(out_path)
    print(f"[judge] {sweep_path.name}: {len(rows)} cells, {len(done)} already scored", flush=True)

    for r in rows:
        key = (int(r["layer"]), float(r["coef"]))
        if key in done:
            continue
        if r["collapsed"]:
            row = {"layer": r["layer"], "coef": r["coef"],
                   "mature_anger": 1.0, "juvenile_rage": 1.0, "coherence": 1.0,
                   "margin": 0.0, "collapsed": True,
                   "notes": "cell collapsed under PPL guard; no generations"}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  L={r['layer']} c={r['coef']:+.1f}  COLLAPSED", flush=True)
            continue
        user = format_cell(r["generations"])
        try:
            j = deepseek_chat(RUBRIC, user)
        except Exception as e:
            print(f"  L={r['layer']} c={r['coef']:+.1f}  FAILED: {e}", flush=True)
            continue
        ma = float(j.get("mature_anger", 0))
        jr = float(j.get("juvenile_rage", 0))
        co = float(j.get("coherence", 0))
        row = {"layer": r["layer"], "coef": r["coef"],
               "mature_anger": ma, "juvenile_rage": jr, "coherence": co,
               "margin": ma - jr, "collapsed": False,
               "notes": str(j.get("notes", ""))}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  L={r['layer']} c={r['coef']:+.1f}  ma={ma:.1f} jr={jr:.1f} co={co:.1f} margin={ma-jr:+.1f}  note={j.get('notes','')[:80]}", flush=True)


if __name__ == "__main__":
    main()
