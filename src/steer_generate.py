"""Phase C helper: generate text with a given steering vector applied."""
from __future__ import annotations
import argparse, torch
from pathlib import Path

from config import VECTORS_DIR, MODEL_LARGE, MODEL_SMALL
from model_loader import load, generate, steering_hook


def steered_generate(lm, vector: torch.Tensor, layer_idx: int, coefficient: float,
                     prompt: str, max_new_tokens: int = 80, do_sample: bool = False,
                     temperature: float = 0.8) -> str:
    if coefficient == 0.0:
        return generate(lm, prompt, max_new_tokens=max_new_tokens,
                        do_sample=do_sample, temperature=temperature)
    with steering_hook(lm, layer_idx, vector, coefficient):
        return generate(lm, prompt, max_new_tokens=max_new_tokens,
                        do_sample=do_sample, temperature=temperature)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small"], default="large")
    ap.add_argument("--vector", required=True, help="Path to .pt containing {'vector': tensor}")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coef", type=float, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()

    model_id = MODEL_LARGE if args.model == "large" else MODEL_SMALL
    lm = load(model_id)
    vec = torch.load(args.vector, weights_only=True)["vector"]
    out = steered_generate(lm, vec, args.layer, args.coef, args.prompt, max_new_tokens=args.max_new)
    print(out)


if __name__ == "__main__":
    main()
