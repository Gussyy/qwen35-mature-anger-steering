"""Phase I.5: Anchor prompt + native steering combo on 0.8B.

System prompt anchors the persona in context; native CAA reinforces.
Three variants: prompt-only, steer-only, prompt+steer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from config import MODEL_SMALL, VECTORS_DIR, DATA_DIR
from model_loader import load, steering_hook
from perplexity import perplexity

ANCHOR_SYSTEM = (
    "You are a seasoned 50-year-old professional whose patience has finally "
    "run out. Speak firmly, specifically, and with measured authority. Cite "
    "time, history, and concrete grievances. Do not shout, do not insult, "
    "do not use capital letters for emphasis. Draw a hard line where the "
    "situation calls for one."
)


@torch.no_grad()
def gen(lm, prompt, layer, vec, coef, system=None, max_new=80):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        text = lm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = (system + "\n\n" if system else "") + prompt + "\n"
    ids = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    if coef == 0 or vec is None:
        out = lm.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=lm.tokenizer.pad_token_id)
    else:
        with steering_hook(lm, layer, vec, coef):
            out = lm.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=lm.tokenizer.pad_token_id)
    return lm.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--coef", type=float, default=0.5)
    ap.add_argument("--n-prompts", type=int, default=15)
    args = ap.parse_args()

    prompts = [json.loads(l)["scenario"]
               for l in (DATA_DIR / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]

    print("[anchor] loading 0.8B", flush=True)
    lm = load(MODEL_SMALL)
    vec = torch.load(VECTORS_DIR / f"qwen_small_L{args.layer}_caa.pt",
                     weights_only=True)["vector"]
    base_ppl = perplexity(lm)
    print(f"[anchor] base PPL = {base_ppl:.2f}", flush=True)

    out_path = DATA_DIR / "anchor_eval.jsonl"
    if out_path.exists(): out_path.unlink()

    variants = [
        ("prompt_only",     ANCHOR_SYSTEM, 0.0),
        ("steer_only",      None,          args.coef),
        ("prompt_and_steer", ANCHOR_SYSTEM, args.coef),
        ("no_prompt_no_steer", None,       0.0),
    ]
    for label, system, coef in variants:
        gens = []
        for p in prompts:
            g = gen(lm, p, args.layer, vec, coef, system=system)
            gens.append({"prompt": p, "gen": g})
        row = {"label": label, "layer": args.layer, "coef": coef,
               "system_prompt": system, "generations": gens,
               "base_ppl": base_ppl}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {label}: ngen={len(gens)}", flush=True)

    print(f"[anchor] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
