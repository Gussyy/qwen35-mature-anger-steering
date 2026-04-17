# LLM Steering — Qwen3.5 Cross-Size Experiment

A home-lab study of **activation steering** on Qwen3.5 models (0.8B, 2B,
4B). We steer toward a "mature + mad" persona using Contrastive
Activation Addition, identify the sparse features responsible via a
custom-trained Top-K SAE, test whether the steering vector can be
transferred cross-size **without any fine-tuning**, and measure how
steering quality changes with model scale.

**Hardware**: single RTX 4070 Ti (12 GB), Windows 11, CUDA 12.4.
**Judges**: Claude + **DeepSeek Chat** (independent API).

> 📘 **New to steering?** [report.md §1](report.md) is a beginner-friendly
> intro — what the residual stream is, how CAA builds a steering vector,
> how SAEs expose the amplified feature, why steering fails on small
> models, and how to tell when it's working. No ML-paper background
> assumed.

## Headline results

### 1. Steering scales cleanly with model size (DeepSeek-judged)

| Model     | Params | Base PPL | Best cell   | mature_anger | juvenile_rage | coherence | Margin | PPL |
|-----------|--------|----------|-------------|--------------|---------------|-----------|--------|-----|
| Qwen3.5-0.8B | 0.8 B | 29.04 | L=6 c=+1.0  | 1.0 | 1.0 | 4.0 | **0.0** | 1.05× |
| Qwen3.5-2B   | 2 B   | 18.94 | L=14 c=+1.0 | 3.5 | 1.0 | 5.0 | **+2.5** | 1.13× |
| Qwen3.5-4B   | 4 B   | 14.33 | L=13 c=+1.0 | **4.5** | 1.0 | 5.0 | **+3.5** | 1.11× |

0.8B cannot host the "mature anger" persona at all. 2B hosts it in a
single cell (L=14, c=+1). 4B hosts it in **four cells** (L=13 c=+1,
L=13 c=+2, L=18 c=+1, L=23 c=+2) with higher margins — the persona
representation is more robust and more broadly accessible at larger
scale. *Thinking mode was disabled for 4B; without that, the model
produces internal chain-of-thought instead of the target persona.*

### 2. SAE analysis: the 2B has a dedicated "mature anger" feature

Feature 4617 in the 2B's L=14 SAE spikes from mean 0.11 → 1.79 (16×)
under steering, fires on 100% of tokens. The 0.8B's strongest shift is
0.16 (10× weaker). The magnitude gap is the mechanistic explanation for
the steering-quality gap.

### 3. Cross-size transfer without training: we tried seven methods

| Method | Approach | DeepSeek margin | Coherence |
|---|---|---|---|
| 0.8B native CAA | own vector | **−2.5** (juvenile) | 2.5 |
| Ridge-projected from 2B | linear map | 0.0 | 4.0 |
| Procrustes-projected from 2B | orthogonal map | 0.0 | 4.5 |
| Random gaussian baseline | same norm | 0.0 | 5.0 |
| **Activation patching** (copy 2B residuals) | ridge or procrustes | 0.0 | 1.0 (gibberish) |
| **SAE-feature steering** (clamp 0.8B features nearest to 2B feat 4617) | 3 alphas | 0.0 | 1.0-1.5 (collapse) |
| **Multi-layer CAA stacking** (all 5 layers simultaneously) | 4 alphas | 0.0 | 1.5 (collapse) |
| **System-prompt anchor only** (no steering) | prompt engineering | **+1.5** ✓ | 4.0 |
| **Prompt + steering combo** | both | +1.0 | 3.0 |

**Only the system-prompt anchor produced positive margin on 0.8B.**
Every residual-level transfer method — linear projection, activation
patching, SAE feature clamping, multi-layer stacking — failed to
induce the persona. Adding native steering on top of the prompt
actively hurt (+1.5 → +1.0) because 0.8B's native CAA direction is
juvenile-coded. The 0.8B clearly *can* produce mature anger in context
(prompt alone works), but no residual-stream intervention can reliably
elicit it.

### 4. Why residual transfer fails: latent-diff analysis

- **Ridge transfer map held-out R² = 0.20** — only 20% of residual
  variance generalizes cross-size; the map overfits.
- **CCA top-9 directions** align at 0.83–0.99 on held-out; the rest
  drop below 0.8. Most of the two models' residual spaces are
  effectively orthogonal.
- **Projected 2B feature 4617 has no near-analog in 0.8B**: best
  cosine 0.26 (ridge) or 0.18 (Procrustes) — noise-level correlations.
- `cos(v_ridge, v_small_native) = +0.32`;
  `cos(v_procrustes, v_small_native) = +0.09`. Procrustes projects to
  a direction nearly orthogonal to what the 0.8B itself identifies
  as "mad".

## Interpretation

Three interlocking conclusions:

1. **Steering is a feature selector, not a feature generator.**
   It amplifies circuits the model already has. No coefficient, layer,
   or alignment map can create a circuit that isn't there.
2. **Cross-size residual transfer is capacity-bounded.** Linear maps
   (ridge, Procrustes) preserve statistics but not features; only ~9
   directions align cross-size and the mature-anger feature is not
   among them. Projecting a feature that exists in the source into a
   target basis that lacks it produces noise.
3. **Prompt conditioning can succeed where steering cannot.** The 0.8B
   can produce mature anger when given a system-prompt anchor — the
   persona IS latent in its weights, but inaccessible from
   residual-stream edits at L=14 alone.

Combined with the scaling trend (0 → +2.5 → +3.5 at 0.8B → 2B → 4B),
the story is that **this persona emerges as a cleanly-steerable feature
somewhere between 0.8B and 2B, and strengthens above 2B**.

## Full detail

Complete writeup — methodology, every training-run metric, every sweep
cell, every SAE feature table, every transfer experiment — in
**[report.md](report.md)**.

## Quick reproduction

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

set PYTHONPATH=src
python src\smoke_test.py
python src\compute_caa_vector.py --model large
python src\compute_caa_vector.py --model small
python src\compute_caa_vector.py --model xlarge --layers 13 18 23
python src\sweep_layers.py --model large
python src\sweep_layers.py --model small
python src\sweep_layers.py --model xlarge --layers 13 18 23 --coefs 1.0 2.0 4.0
python src\train_sae.py --model large --layer 14 --tokens 1500000
python src\train_sae.py --model small --layer 14 --tokens 1500000
python src\sae_analysis.py --model large --layer 14 --coef 1.0
python src\sae_analysis.py --model small --layer 14 --coef 1.0
python src\transfer_vector.py --l-large 14 --l-small 14 --c-small 1.0
python src\latent_diff_analysis.py
python src\activation_patching.py
python src\sae_feature_steer.py --feat-ids 79 973 1206 1367 137 --alphas 1.0 2.0 4.0
python src\multi_layer_steer.py --alphas 0.5 1.0 1.5 2.0
python src\anchor_plus_steer.py --coef 0.5
python src\deepseek_judge.py --sweep data\sweep_qwen_large.jsonl --out data\judge_deepseek_large.jsonl
python src\deepseek_judge.py --sweep data\sweep_qwen_small.jsonl --out data\judge_deepseek_small.jsonl
python src\deepseek_judge.py --sweep data\sweep_qwen_xlarge.jsonl --out data\judge_deepseek_xlarge.jsonl
python src\judge_phase_i.py
```

`.env` must contain `huggingface_hub_token=...` and `DEEPSEEK_KEY=...`.
End-to-end runtime: **≈ 120 minutes** on a 4070 Ti. All intermediate
artifacts land on disk; any phase can be rerun independently.

## Repo map

```
src/                         14 pipeline + judging scripts
data/
  contrast_pairs.jsonl       80 mature_mad / neutral pairs
  eval_prompts.jsonl         40 held-out scenarios
  sweep_qwen_{large,small,xlarge}.jsonl   all generations in the sweep
  judge_qwen_{large,small}.jsonl          Claude self-judge scores
  judge_deepseek_{large,small,xlarge}.jsonl  DeepSeek independent scores
  judge_transfer.jsonl / judge_deepseek_transfer.jsonl    transfer scores
  patch_eval.jsonl, sae_steer_eval.jsonl,
    multi_layer_eval.jsonl, anchor_eval.jsonl   Phase I generations
  judge_phase_i.jsonl        unified Phase I DeepSeek scores
  transfer_eval.jsonl        Phase G transfer-variant generations
  size_comparison.json       programmatic cross-size summary
  latent_diff.json           alignment diagnostics
vectors/
  qwen_{large,small,xlarge}_L*_caa.pt   14 per-layer CAA vectors
  transfer_map_large_to_small.pt        ridge + Procrustes maps
  qwen_small_transferred_caa.pt         projected vectors
saes/
  qwen_{large,small}_L14_sae.pt         trained Top-K SAEs
  qwen_{large,small}_L14_features.json  top-30 feature deltas
```

See **[report.md](report.md)** for full analysis including the scaling
trend, Phase I no-training-transfer experiments, and judge comparison.
