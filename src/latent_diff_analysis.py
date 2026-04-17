"""Phase I.1: How different are 2B and 0.8B's L=14 latent spaces really?

We measure four things:
  1. CCA correlation spectrum between residual streams on 30 shared prompts
     (how many directions align well, and how quickly does correlation decay?).
  2. Ridge-transfer held-out R^2 on 20 fresh prompts (does the map generalize?).
  3. Cosine between (a) the 2B CAA vector projected via ridge/Procrustes and
     (b) the 0.8B native CAA vector. If these are orthogonal, transfer is
     fundamentally pointing in a different direction from what 0.8B calls
     'mad'.
  4. Where in the 0.8B SAE the 2B's top feature 4617 direction projects to
     (which 0.8B features does it correlate with on shared text?).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge

from config import MODEL_LARGE, MODEL_SMALL, VECTORS_DIR, SAES_DIR, DATA_DIR
from model_loader import load, capture_residual
from train_sae import TopKSAE


NEUTRAL_PROMPTS = [
    "Preheat the oven to 350 degrees and prepare the baking sheet.",
    "The stock market closed slightly higher today after a volatile session.",
    "Photosynthesis converts sunlight and carbon dioxide into glucose.",
    "The conference room is available from 2 to 3 on Thursday afternoon.",
    "A square has four equal sides and four right angles.",
    "Tokyo is the capital of Japan and one of the world's largest cities.",
    "Please submit your timesheet by 5 PM on Friday.",
    "The novel was published in 1953 and won several literary awards.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The new library opens at nine on weekdays and ten on Saturdays.",
    "Mount Everest stands at approximately 8,849 meters above sea level.",
    "The recipe calls for two cups of flour and one teaspoon of salt.",
    "Photons are elementary particles that transmit electromagnetic force.",
    "The committee meets on the second Tuesday of each month.",
    "The Pacific Ocean is the largest of the world's five oceans.",
    "Please cite three sources to support your argument.",
    "Gravity on Earth is approximately 9.8 meters per second squared.",
    "The museum is hosting an exhibition on medieval manuscripts this fall.",
    "The algorithm runs in O(n log n) time on average.",
    "The president delivered a short address on infrastructure spending.",
    "Every triangle has interior angles that sum to 180 degrees.",
    "The restaurant specializes in regional cuisine and fresh seafood.",
    "The bookstore discounts hardcovers by 20 percent on weekends.",
    "The train leaves from platform seven at a quarter past four.",
    "A decimal point separates the whole-number part from the fractional part.",
    "The new policy takes effect at the start of the fiscal year.",
    "Photographs from the expedition will be published next month.",
    "The annual festival draws visitors from across the region.",
    "Mitochondria generate energy through oxidative phosphorylation.",
    "The quarterly report is available on the company website.",
]


def mean_prompt_resid(lm, layer_idx, text):
    try:
        prompt_text = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = text + "\n"
    ids = lm.tokenizer(prompt_text, return_tensors="pt").to(lm.model.device)
    collected = []
    with capture_residual(lm, layer_idx, collected), torch.no_grad():
        lm.model(**ids)
    return collected[0][0].float().cpu().mean(dim=0)


def collect_matrix(model_id, layer, prompts):
    lm = load(model_id)
    rows = [mean_prompt_resid(lm, layer, p).numpy() for p in prompts]
    del lm; torch.cuda.empty_cache()
    import gc; gc.collect()
    return np.stack(rows)


def main():
    # Held-out prompts for generalization
    fit_prompts = NEUTRAL_PROMPTS[:20]
    held_prompts = NEUTRAL_PROMPTS[20:]  # 10 held-out

    print("[diff] collecting 2B residuals at L=14", flush=True)
    X_fit = collect_matrix(MODEL_LARGE, 14, fit_prompts)
    X_hel = collect_matrix(MODEL_LARGE, 14, held_prompts)
    print("[diff] collecting 0.8B residuals at L=14", flush=True)
    Y_fit = collect_matrix(MODEL_SMALL, 14, fit_prompts)
    Y_hel = collect_matrix(MODEL_SMALL, 14, held_prompts)

    # --- (1) Ridge held-out R^2 ---
    ridge = Ridge(alpha=1.0).fit(X_fit, Y_fit)
    Y_pred = ridge.predict(X_hel)
    ss_res = ((Y_hel - Y_pred) ** 2).sum()
    ss_tot = ((Y_hel - Y_hel.mean(axis=0)) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"[diff] ridge held-out R^2 = {r2:.4f} (alpha=1.0)", flush=True)

    # --- (2) CCA correlation spectrum ---
    n_comp = min(20, X_fit.shape[0])
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(X_fit, Y_fit)
    Xc, Yc = cca.transform(X_fit, Y_fit)
    corrs = []
    for i in range(n_comp):
        corrs.append(float(np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]))
    print(f"[diff] CCA top-{n_comp} correlations (fit): {[round(c,3) for c in corrs]}", flush=True)
    Xh, Yh = cca.transform(X_hel, Y_hel)
    hold_corrs = [float(np.corrcoef(Xh[:, i], Yh[:, i])[0, 1]) for i in range(n_comp)]
    print(f"[diff] CCA top-{n_comp} correlations (held-out): {[round(c,3) for c in hold_corrs]}", flush=True)

    # --- (3) Cosines between transferred CAA vector and 0.8B native ---
    v_large = torch.load(VECTORS_DIR / "qwen_large_L14_caa.pt", weights_only=True)["vector"].float().numpy()
    v_small = torch.load(VECTORS_DIR / "qwen_small_L14_caa.pt", weights_only=True)["vector"].float().numpy()
    tm = torch.load(VECTORS_DIR / "transfer_map_large_to_small.pt", weights_only=False)
    W, R = tm["W"], tm["R"]
    v_ridge = v_large @ W
    v_proc  = R @ v_large
    def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[diff] cos( v_ridge, v_small_native ) = {cos(v_ridge, v_small):+.4f}", flush=True)
    print(f"[diff] cos( v_proc,  v_small_native ) = {cos(v_proc, v_small):+.4f}", flush=True)
    print(f"[diff] cos( v_ridge, v_proc )         = {cos(v_ridge, v_proc):+.4f}", flush=True)

    # --- (4) Where does 2B feature 4617 project in 0.8B's SAE? ---
    # Load both SAEs
    def load_sae(p, dev):
        ck = torch.load(p, weights_only=True)
        sae = TopKSAE(ck["d_model"], ck["d_sae"], ck["k"]).to(dev).float()
        sae.load_state_dict(ck["state_dict"])
        sae.eval()
        return sae, ck
    # We only need the direction of feature 4617 from 2B SAE: the column of W_enc (or row of W_dec).
    sae2b_ck = torch.load(SAES_DIR / "qwen_large_L14_sae.pt", weights_only=True)
    W_dec_2b = sae2b_ck["state_dict"]["W_dec"].float().cpu().numpy()  # (d_sae, d_model)
    feat_4617_dir_2b = W_dec_2b[4617]  # (2048,)
    # Project it through ridge into 0.8B space
    f_ridge = feat_4617_dir_2b @ W
    f_proc  = R @ feat_4617_dir_2b
    print(f"[diff] 2B feat 4617 decoder vec norm: {np.linalg.norm(feat_4617_dir_2b):.3f}", flush=True)
    print(f"[diff] ridge-projected to 0.8B norm: {np.linalg.norm(f_ridge):.3f}", flush=True)
    print(f"[diff] proc-projected  to 0.8B norm: {np.linalg.norm(f_proc):.3f}", flush=True)

    # Compare to 0.8B SAE decoder rows — which 0.8B features is 4617 most aligned with?
    sae08_ck = torch.load(SAES_DIR / "qwen_small_L14_sae.pt", weights_only=True)
    W_dec_08 = sae08_ck["state_dict"]["W_dec"].float().cpu().numpy()  # (d_sae, 1024)
    # Cosine similarity between f_ridge (in 0.8B space) and each 0.8B decoder direction
    def norm_rows(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    W_dec_08_n = norm_rows(W_dec_08)
    for v, label in [(f_ridge, "ridge-projected"), (f_proc, "proc-projected")]:
        v_n = v / (np.linalg.norm(v) + 1e-9)
        sims = W_dec_08_n @ v_n  # (d_sae,)
        top_k = np.argsort(-np.abs(sims))[:10]
        print(f"[diff] {label} — top 10 closest 0.8B SAE features by cosine:", flush=True)
        for i in top_k:
            print(f"         feat {int(i):>5}  cos={sims[i]:+.4f}", flush=True)

    # Save summary
    out = DATA_DIR / "latent_diff.json"
    out.write_text(json.dumps({
        "ridge_heldout_r2": r2,
        "cca_fit_corrs": corrs,
        "cca_heldout_corrs": hold_corrs,
        "cos_ridge_native": cos(v_ridge, v_small),
        "cos_proc_native": cos(v_proc, v_small),
        "cos_ridge_proc": cos(v_ridge, v_proc),
        "feat_4617_ridge_norm": float(np.linalg.norm(f_ridge)),
        "feat_4617_proc_norm": float(np.linalg.norm(f_proc)),
    }, indent=2), encoding="utf-8")
    print(f"[diff] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
