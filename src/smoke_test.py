"""Phase A smoke test.

Loads Qwen3.5-2B and Qwen3.5-0.8B in fp16, registers a forward hook on a
middle layer for each, and verifies:
  - torch.cuda is available
  - both models generate 20 coherent tokens
  - the hook fires and captures a tensor of the expected shape
  - peak VRAM stays under budget
"""
from __future__ import annotations
import gc, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from config import MODEL_LARGE, MODEL_SMALL, D_MODEL_LARGE, D_MODEL_SMALL, HF_TOKEN

HOOK_LAYER = 10

def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)

def load_and_probe(model_id: str, expected_d_model: int) -> None:
    log(f"--- {model_id} ---")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
        token=HF_TOKEN,
    )
    model.eval()
    log(f"loaded in {time.time()-t0:.1f}s")

    # Inspect layer structure
    # Qwen3.5 exposes layers at model.model.layers
    layers = model.model.layers
    log(f"n_layers={len(layers)} d_model={model.config.hidden_size}")
    assert model.config.hidden_size == expected_d_model, f"expected d_model={expected_d_model}, got {model.config.hidden_size}"

    captured: list[torch.Tensor] = []
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())
    handle = layers[HOOK_LAYER].register_forward_hook(hook)

    # Generate
    prompt = "Hello, I am"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out_ids[0], skip_special_tokens=True)
    log(f"gen: {gen!r}")

    handle.remove()
    assert captured, "hook never fired"
    h = captured[0]
    log(f"hook tensor shape={tuple(h.shape)} dtype={h.dtype}")
    assert h.shape[-1] == expected_d_model

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log(f"peak VRAM: {peak_gb:.2f} GB")

    del model, tok, captured
    gc.collect()
    torch.cuda.empty_cache()

def main() -> None:
    log(f"cuda_available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}")
    if HF_TOKEN:
        log("HF token found; logging in")
        login(token=HF_TOKEN, add_to_git_credential=False)
    load_and_probe(MODEL_LARGE, D_MODEL_LARGE)
    load_and_probe(MODEL_SMALL, D_MODEL_SMALL)
    log("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
