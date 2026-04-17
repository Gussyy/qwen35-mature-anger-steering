"""Phase E: train a small Top-K Sparse Autoencoder on residual-stream
activations of one model at one layer.

Minimal, self-contained TopK SAE — we don't use sae_lens' runner because
Qwen3.5's hybrid architecture doesn't play with HookedTransformer.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Iterator, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from config import MODEL_LARGE, MODEL_SMALL, SAES_DIR
from model_loader import load, capture_residual


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * (1.0 / math.sqrt(d_model)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(self.W_enc.detach().clone().T)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, F.relu(vals))
        return acts, pre

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts, pre = self.encode(x)
        x_hat = self.decode(acts)
        return x_hat, acts, pre


def stream_text(tokenizer, batch_size: int, seq_len: int) -> Iterator[torch.Tensor]:
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    batch: list[list[int]] = []
    buf: list[int] = []
    for ex in ds:
        ids = tokenizer(ex["text"], add_special_tokens=False, truncation=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []


def collect_activations(lm, layer_idx: int, tokens: torch.Tensor) -> torch.Tensor:
    tokens = tokens.to(lm.model.device)
    collected: list[torch.Tensor] = []
    with capture_residual(lm, layer_idx, collected), torch.no_grad():
        lm.model(tokens)
    resid = collected[0]  # (B, S, d_model)
    return resid.reshape(-1, resid.shape[-1]).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["large", "small"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--d-sae", type=int, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=2_000_000, help="Target training tokens")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-coef", type=float, default=0.0)
    args = ap.parse_args()

    model_id = MODEL_LARGE if args.model == "large" else MODEL_SMALL
    model_tag = "qwen_large" if args.model == "large" else "qwen_small"
    print(f"[sae] loading {model_id}", flush=True)
    lm = load(model_id)
    d_model = lm.d_model
    d_sae = args.d_sae or (4 * d_model)
    print(f"[sae] d_model={d_model} d_sae={d_sae} k={args.k}", flush=True)

    sae = TopKSAE(d_model, d_sae, args.k).to(lm.model.device).float()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    tokens_per_step = args.batch_size * args.seq_len
    target_steps = max(1, args.tokens // tokens_per_step)
    print(f"[sae] target {args.tokens:,} tokens = {target_steps} steps of {tokens_per_step} tokens", flush=True)

    stream = stream_text(lm.tokenizer, args.batch_size, args.seq_len)
    losses: list[float] = []
    t0 = time.time()
    try:
        for step in range(target_steps):
            toks = next(stream)
            acts = collect_activations(lm, args.layer, toks)  # (B*S, d_model)
            x_hat, a, _ = sae(acts)
            recon = F.mse_loss(x_hat, acts)
            loss = recon
            if args.aux_coef > 0:
                # small activation penalty to encourage feature sparsity beyond TopK
                loss = loss + args.aux_coef * a.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(recon))
            if step % 50 == 0:
                var = acts.var().item()
                fvu = float(recon) / max(var, 1e-8)
                print(f"[sae] step {step}/{target_steps} recon={float(recon):.4f} "
                      f"fvu={fvu:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    except StopIteration:
        print("[sae] dataset ended early", flush=True)

    out_path = SAES_DIR / f"{model_tag}_L{args.layer}_sae.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "d_model": d_model, "d_sae": d_sae, "k": args.k,
        "layer": args.layer, "model_id": model_id,
        "final_recon": losses[-1] if losses else None,
        "steps": len(losses),
    }, out_path)
    print(f"[sae] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
