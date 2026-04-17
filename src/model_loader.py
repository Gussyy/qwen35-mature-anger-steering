"""Thin wrapper around HF transformers that exposes the forward-hook based
residual-stream interface we use for CAA + steering. Architecture-agnostic
so we don't depend on TransformerLens (which doesn't support Qwen3.5's
Gated DeltaNet hybrid)."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HF_TOKEN


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    d_model: int
    n_layers: int
    model_id: str

    @property
    def layers(self):
        return self.model.model.layers


def load(model_id: str, dtype=torch.bfloat16) -> LoadedModel:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map="cuda",
        token=HF_TOKEN,
    )
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tok,
        d_model=model.config.hidden_size,
        n_layers=model.config.num_hidden_layers,
        model_id=model_id,
    )


def _resid_from_out(out):
    """Qwen3.5 layer forward returns a tuple (hidden_states, ...)."""
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def capture_residual(lm: LoadedModel, layer_idx: int, collect: list) -> Iterator[None]:
    """Context manager that collects residual tensors from the given layer.

    Each forward pass appends a tensor of shape (batch, seq, d_model) to `collect`.
    """
    def hook(_m, _inp, out):
        collect.append(_resid_from_out(out).detach())
    h = lm.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def steering_hook(
    lm: LoadedModel,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
) -> Iterator[None]:
    """Adds coefficient * vector to the residual stream at layer_idx on every
    forward pass. Vector must be shape (d_model,) on same device/dtype."""
    v = vector.to(device=lm.model.device, dtype=next(lm.model.parameters()).dtype)
    c = float(coefficient)

    def hook(_m, _inp, out):
        resid = _resid_from_out(out)
        resid = resid + c * v
        if isinstance(out, tuple):
            return (resid,) + out[1:]
        return resid

    h = lm.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def generate(
    lm: LoadedModel,
    prompt: str,
    max_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: float = 0.7,
    system: Optional[str] = None,
) -> str:
    """Apply chat template and generate. Returns only the generated continuation."""
    if system is not None:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    try:
        try:
            text = lm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = lm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    except Exception:
        # Fallback if no chat template
        text = prompt + "\n"
    inputs = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    with torch.no_grad():
        out = lm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return lm.tokenizer.decode(gen_ids, skip_special_tokens=True)
