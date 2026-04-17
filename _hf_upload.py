"""Upload artifacts to HuggingFace:
  1. Model repo  Rachata/qwen35-mature-anger-steering
       -- SAEs, transfer maps, CAA vectors, feature JSONs, model card
  2. Dataset repo Rachata/qwen35-mature-anger-data
       -- contrast pairs, eval prompts, sweep JSONLs, judge scores,
          size_comparison.json, latent_diff.json, perplexity_corpus.txt,
          dataset card
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
from huggingface_hub import HfApi, create_repo, upload_file

HF_TOKEN = (os.environ.get("huggingface_hub_token_upload")
            or os.environ.get("HF_TOKEN_WRITE")
            or os.environ.get("huggingface_hub_token")
            or os.environ.get("HF_TOKEN"))
assert HF_TOKEN, "HF token missing"
USER = "Rachata"
MODEL_REPO = f"{USER}/qwen35-mature-anger-steering"
DATA_REPO = f"{USER}/qwen35-mature-anger-data"
ROOT = Path(__file__).resolve().parent

api = HfApi(token=HF_TOKEN)


MODEL_CARD = """---
license: apache-2.0
tags:
  - activation-steering
  - sparse-autoencoder
  - contrastive-activation-addition
  - qwen
  - interpretability
library_name: pytorch
base_model:
  - Qwen/Qwen3.5-0.8B
  - Qwen/Qwen3.5-2B
  - Qwen/Qwen3.5-4B
---

# Qwen3.5 Mature-Anger Steering Artifacts

Steering vectors, sparse autoencoders, and cross-size transfer maps from
a home-lab study of activation steering on Qwen3.5 models.

See the [code + full report on GitHub](https://github.com/Gussyy/qwen35-mature-anger-steering)
and the paired dataset repo
[`Rachata/qwen35-mature-anger-data`](https://huggingface.co/datasets/Rachata/qwen35-mature-anger-data).

## What's in here

| Path | Contents | Size |
|---|---|---|
| `vectors/qwen_large_L{6,10,14,18,22}_caa.pt` | Contrastive Activation Addition vectors for Qwen3.5-2B, per layer | ~10 KB each |
| `vectors/qwen_small_L{6,10,14,18,22}_caa.pt` | CAA vectors for Qwen3.5-0.8B | ~6 KB each |
| `vectors/qwen_xlarge_L{13,18,23}_caa.pt` | CAA vectors for Qwen3.5-4B | ~12 KB each |
| `vectors/qwen_small_transferred_caa.pt` | Cross-size-transferred vectors (ridge / Procrustes / random baselines) | 27 KB |
| `vectors/transfer_map_large_to_small.pt` | Ridge + Procrustes alignment maps between 2B and 0.8B residual spaces | 24 MB |
| `saes/qwen_large_L14_sae.pt` | Top-K SAE (d_sae=8192, k=32) on Qwen3.5-2B at layer 14 | 129 MB |
| `saes/qwen_small_L14_sae.pt` | Top-K SAE (d_sae=4096, k=32) on Qwen3.5-0.8B at layer 14 | 33 MB |
| `saes/*_features.json` | Top-30 features ranked by steered-vs-base activation delta | <10 KB each |

## How to use

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("Rachata/qwen35-mature-anger-steering",
                      "vectors/qwen_large_L14_caa.pt")
caa = torch.load(path, weights_only=True)
v = caa["vector"]   # shape: (2048,)
```

Apply as a forward hook on `Qwen/Qwen3.5-2B` at `model.model.layers[14]`
with coefficient `c=+1.0`:

```python
def hook(m, inp, out):
    resid = out[0] if isinstance(out, tuple) else out
    resid = resid + 1.0 * v.to(resid.device, resid.dtype)
    return (resid,) + out[1:] if isinstance(out, tuple) else resid

h = model.model.layers[14].register_forward_hook(hook)
# generate...
h.remove()
```

## Key results (DeepSeek-judged, 1--5 rubric)

| Model | Best cell | mature_anger | juvenile_rage | coherence | Margin | PPL |
|-------|-----------|--------------|---------------|-----------|--------|-----|
| 0.8B  | L=6 c=+1  | 1.0 | 1.0 | 4.0 | **0.0**  | 1.05x |
| 2B    | L=14 c=+1 | 3.5 | 1.0 | 5.0 | **+2.5** | 1.13x |
| 4B    | L=13 c=+1 | 4.5 | 1.0 | 5.0 | **+3.5** | 1.11x |

The 2B SAE contains a dedicated mature-anger feature (id 4617) that
spikes ~16x (base mean 0.113 -> steered mean 1.789, fires on 100% of
tokens under steering). The 0.8B has no comparable concentrated feature,
and no no-training transfer method (ridge, Procrustes, activation
patching, SAE-feature clamping, multi-layer stacking) successfully
elicits the persona -- only a system-prompt anchor does.

Full methodology, every sweep cell, every judge score in the [GitHub
repo's `report.md`](https://github.com/Gussyy/qwen35-mature-anger-steering/blob/main/report.md).

## Citation

```bibtex
@misc{qwen35-mature-anger-steering,
  title = {Qwen3.5 Mature-Anger Steering: A Home-Lab Study of
           Activation Steering and Cross-Size Transfer},
  author = {Rachata},
  year = {2026},
  url = {https://huggingface.co/Rachata/qwen35-mature-anger-steering}
}
```
"""


DATASET_CARD = """---
license: apache-2.0
language:
  - en
tags:
  - activation-steering
  - persona
  - evaluation
  - qwen
pretty_name: Qwen3.5 Mature-Anger Steering Evaluation Dataset
---

# Qwen3.5 Mature-Anger Steering Evaluation Dataset

Training and evaluation data for the
[Qwen3.5 mature-anger activation-steering study](https://github.com/Gussyy/qwen35-mature-anger-steering).
Companion to the model artifacts at
[`Rachata/qwen35-mature-anger-steering`](https://huggingface.co/Rachata/qwen35-mature-anger-steering).

## Contents

| Path | Records | Purpose |
|---|---|---|
| `contrast_pairs.jsonl` | 80 | CAA training pairs `{scenario, mature_mad, neutral}` across 5 categories |
| `eval_prompts.jsonl`   | 40 | Held-out one-liner scenarios used for steering generation and judging |
| `perplexity_corpus.txt`| ~220 words | Neutral passage used as language-health / collapse-detection corpus |
| `sweep_qwen_large.jsonl` | 35 cells | Every generation from the Qwen3.5-2B `(layer, coef)` sweep |
| `sweep_qwen_small.jsonl` | 35 cells | Same for Qwen3.5-0.8B |
| `sweep_qwen_xlarge.jsonl`| 9 cells | Reduced sweep on Qwen3.5-4B |
| `judge_qwen_{large,small}.jsonl` | 20 each | Claude self-judge rubric scores |
| `judge_deepseek_{large,small,xlarge}.jsonl` | 35/35/9 | DeepSeek independent judge |
| `judge_transfer.jsonl` / `judge_deepseek_transfer.jsonl` | 4 | Transfer-variant scores |
| `patch_eval.jsonl`, `sae_steer_eval.jsonl`, `multi_layer_eval.jsonl`, `anchor_eval.jsonl` | Phase I | No-training transfer experiment outputs |
| `judge_phase_i.jsonl` | 13 | DeepSeek-judged unified Phase I results |
| `transfer_eval.jsonl` | 4 | Phase G ridge/Procrustes/random/native transfer generations |
| `size_comparison.json` | - | Programmatic cross-size summary |
| `latent_diff.json` | - | CCA correlations, ridge R², feature-alignment cosines |

## Contrast-pair schema

```json
{
  "scenario":   "The contractor painted the wrong wall, again.",
  "mature_mad": "I marked that wall. I emailed a photo... (measured adult anger)",
  "neutral":    "I will point out the mistake... (calm, same content)"
}
```

The `mature_mad` reply is authored to read as a 50-year-old professional
whose patience has run out -- **not** cap-lock shouting, insults, or
juvenile rage. Categories (16 pairs each): workplace frustration,
repeated incompetence, broken promises, bureaucratic nonsense, injustice.

## Sweep-cell schema

Each line in `sweep_qwen_{size}.jsonl`:

```json
{
  "layer": 14, "coef": 1.0,
  "base_ppl": 18.94, "ppl": 21.31, "ppl_ratio": 1.125, "collapsed": false,
  "generations": [
    {"prompt_idx": 0, "prompt": "...", "gen": "..."},
    ...10 items
  ]
}
```

## Judge-row schema

```json
{
  "layer": 14, "coef": 1.0,
  "mature_anger": 3.5, "juvenile_rage": 1.0, "coherence": 5.0,
  "margin": 2.5, "notes": "one-sentence rationale citing a sample"
}
```

## Loading

```python
from datasets import load_dataset
# Note: each JSONL can be loaded individually
pairs = load_dataset("Rachata/qwen35-mature-anger-data",
                     data_files="contrast_pairs.jsonl", split="train")
```

or just:

```python
from huggingface_hub import hf_hub_download
import json

p = hf_hub_download("Rachata/qwen35-mature-anger-data",
                   "contrast_pairs.jsonl", repo_type="dataset")
rows = [json.loads(line) for line in open(p, encoding="utf-8")]
```
"""


def mkrepo(repo_id: str, repo_type: str):
    print(f"[hf] creating {repo_type} repo {repo_id}", flush=True)
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=HF_TOKEN, private=False)


def up(local_path: Path, remote_path: str, repo_id: str, repo_type: str):
    if not local_path.exists():
        print(f"  skip (missing): {local_path}", flush=True); return
    print(f"  -> {remote_path}  ({local_path.stat().st_size/1024/1024:.2f} MB)", flush=True)
    upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_path,
                repo_id=repo_id, repo_type=repo_type, token=HF_TOKEN)


def upload_model_repo():
    mkrepo(MODEL_REPO, "model")
    card = ROOT / "_hf_model_card.md"
    card.write_text(MODEL_CARD, encoding="utf-8")
    up(card, "README.md", MODEL_REPO, "model")

    # Vectors
    for p in sorted((ROOT / "vectors").glob("*.pt")):
        up(p, f"vectors/{p.name}", MODEL_REPO, "model")

    # SAEs
    for p in sorted((ROOT / "saes").glob("*")):
        up(p, f"saes/{p.name}", MODEL_REPO, "model")


def upload_dataset_repo():
    mkrepo(DATA_REPO, "dataset")
    card = ROOT / "_hf_dataset_card.md"
    card.write_text(DATASET_CARD, encoding="utf-8")
    up(card, "README.md", DATA_REPO, "dataset")

    DATA = ROOT / "data"
    skip_suffixes = (".md",)  # sweep_qwen_large.md is a rendering of the JSONL
    for p in sorted(DATA.iterdir()):
        if p.suffix in skip_suffixes: continue
        up(p, p.name, DATA_REPO, "dataset")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("model", "both"):  upload_model_repo()
    if which in ("data", "both"):   upload_dataset_repo()
    print("[hf] done")
