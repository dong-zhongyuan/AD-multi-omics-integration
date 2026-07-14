#!/usr/bin/env python3
"""
Cross-species transfer validation for the Neural ODE world model.

Loads the pre-trained 5xFAD mouse world model, evaluates it on human ADNI
bulk transcriptome data (same ortholog-mapped gene space), then fine-tunes
for a small number of epochs and reports the OT-distance improvement.

This is a POST-HOC validation script — it does NOT modify any existing
pipeline step, result, or figure. It only reads:
  - saved model checkpoint (output/step1_world_model_transcriptomics_no_pca/checkpoints/best.pt)
  - 5xFAD training data gene list (processed-data/transcriptomics_blood.h5ad var_names)
  - ADNI human expression (data/survival/ADNI_Gene_Expression_Profile.csv)
  - DXSUM diagnosis (data/blood-transcription-protein/DXSUM_17Apr2026.csv)

Output: output/step1_world_model_transcriptomics_no_pca/cross_species_transfer.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT, "tools", "hepaworld"))
from models.dynamics import DriftNet, integrate_ode

CKPT = os.path.join(PROJECT, "output/step1_world_model_transcriptomics_no_pca/checkpoints/best.pt")
FIVEXFAD_BLOOD = os.path.join(PROJECT, "processed-data/transcriptomics_blood.h5ad")
FIVEXFAD_BRAIN = os.path.join(PROJECT, "processed-data/transcriptomics_brain.h5ad")
ADNI_EXPR = os.path.join(PROJECT, "data/survival/ADNI_Gene_Expression_Profile.csv")
OUT_JSON = os.path.join(PROJECT, "output/step1_world_model_transcriptomics_no_pca/cross_species_transfer.json")

# ---- 1. Load model ----
print("Loading model checkpoint...")
ckpt = torch.load(CKPT, map_location='cpu')
cfg = ckpt['cfg']
gene_dim = cfg['gene_dim']  # 2000
model = DriftNet(dim=gene_dim, hidden=64, depth=2, time_freq=4, dropout=0.0)
# The saved model uses depth=2 which creates layers [Linear, ReLU, Linear] indexed as mlp.0, mlp.1(relu), mlp.2
# But saved state_dict has mlp.0 and mlp.3 — meaning depth=2 with dropout creates [Linear, ReLU, Dropout, Linear]
# Match by loading with strict=False first to diagnose
try:
    model.load_state_dict(ckpt['state_dict'])
except RuntimeError:
    # Rebuild with dropout to match saved architecture
    model = DriftNet(dim=gene_dim, hidden=64, depth=2, time_freq=4, dropout=0.1)
    model.load_state_dict(ckpt['state_dict'])
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"  gene_dim={gene_dim}, hidden=64, depth=2")

# ---- 2. Get 5xFAD gene list (the 2000 HVGs used in training) ----
print("Loading 5xFAD gene space...")
import anndata as ad
adata_blood = ad.read_h5ad(FIVEXFAD_BLOOD, backed='r')
adata_brain = ad.read_h5ad(FIVEXFAD_BRAIN, backed='r')
common_genes = sorted(set(adata_blood.var_names) & set(adata_brain.var_names))
print(f"  5xFAD common genes: {len(common_genes)}")

# ---- 3. Parse ADNI human expression, match to 5xFAD gene space ----
print("Loading ADNI human expression (~30s, 222MB)...")
raw = pd.read_csv(ADNI_EXPR, header=None, low_memory=False)
ptids = raw.iloc[2, 3:].astype(str).tolist()
symbols = raw.iloc[9:, 2].astype(str).reset_index(drop=True)
data = raw.iloc[9:, 3:].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
data.columns = ptids
data["Symbol"] = symbols

# collapse multi-probe per gene (median, skipna)
gmat = data.groupby("Symbol").median(numeric_only=True)
gmat = gmat.fillna(gmat.median())  # fill remaining NaN with column median
print(f"  ADNI genes: {len(gmat)}, subjects: {gmat.shape[1]}")

# match to 5xFAD genes
adni_genes_in_model = [g for g in common_genes if g in gmat.index]
print(f"  ADNI genes matching 5xFAD model: {len(adni_genes_in_model)} / {len(common_genes)}")

# build human expression matrix in model gene space
adni_subset = np.zeros((len(common_genes), gmat.shape[1]), dtype=np.float32)
for i, g in enumerate(common_genes):
    if g in gmat.index:
        vals = gmat.loc[g].values.astype(np.float32)
        # fill NaN with gene mean
        nan_mask = np.isnan(vals)
        if nan_mask.all():
            vals[:] = 0.0
        else:
            vals[nan_mask] = np.nanmean(vals)
        adni_subset[i] = vals

# z-score per gene (across subjects) to match 5xFAD normalized scale
adni_mean = adni_subset.mean(axis=1, keepdims=True)
adni_std = adni_subset.std(axis=1, keepdims=True)
adni_std[adni_std < 1e-6] = 1.0
adni_subset = (adni_subset - adni_mean) / adni_std
adni_subset = np.nan_to_num(adni_subset, nan=0.0, posinf=0.0, neginf=0.0)
adni_tensor = torch.from_numpy(adni_subset.T)  # (n_subjects, gene_dim)
print(f"  ADNI tensor (z-scored): {adni_tensor.shape}, range: [{adni_tensor.min():.2f}, {adni_tensor.max():.2f}]")

# ---- 4. Also get 5xFAD mouse data for comparison ----
print("Loading 5xFAD mouse data for baseline...")
import scipy.sparse as sp
def to_dense(X, n=500):
    block = X[:n]
    if sp.issparse(block):
        block = block.toarray()
    return torch.from_numpy(np.asarray(block, dtype=np.float32))

mouse_blood = to_dense(adata_blood.X, 500)
mouse_brain = to_dense(adata_brain.X, 500)

# ---- 5. Sinkhorn OT distance function (same as training) ----
def ot_distance(x, y, epsilon=0.15, n_iters=50):
    """Sinkhorn OT loss between two batches."""
    C = torch.cdist(x, y, p=2.0) ** 2
    n, m = C.shape
    if n == 0 or m == 0:
        return torch.tensor(0.0)
    mu = torch.full((n,), 1.0/n)
    nu = torch.full((m,), 1.0/m)
    K = torch.exp(-C / epsilon).clamp(min=1e-8)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.t() @ u + 1e-8)
    T = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(T * C)

# ---- 6. Evaluate: mouse OT (baseline) and human OT (pre-finetune) ----
print("\n=== Cross-species transfer evaluation ===")
B = 128  # batch size

# Mouse baseline: blood -> model -> predicted brain, OT vs real brain
idx = np.random.choice(len(mouse_blood), B, replace=False)
x0_mouse = mouse_blood[idx]
with torch.no_grad():
    pred_brain_mouse = integrate_ode(model, x0_mouse, torch.tensor(0.0), torch.tensor(1.0), n_steps=4)
    real_brain_idx = np.random.choice(len(mouse_brain), B, replace=False)
    real_brain_mouse = mouse_brain[real_brain_idx]
    ot_mouse = ot_distance(pred_brain_mouse, real_brain_mouse)
print(f"  Mouse OT (blood→brain, trained): {ot_mouse.item():.2f}")

# Human: blood -> model -> predicted brain (but we have no human brain single-cell to compare)
# Instead, measure: human blood -> model -> output, how far is the output from human blood?
# (a well-transferred model should produce a distinct brain-like distribution)
idx_h = np.random.choice(len(adni_tensor), B, replace=len(adni_tensor) < B)
x0_human = adni_tensor[idx_h]
with torch.no_grad():
    pred_human = integrate_ode(model, x0_human, torch.tensor(0.0), torch.tensor(1.0), n_steps=4)
    # OT between input and output — lower means model is "doing something" (transforming)
    ot_human_pre = ot_distance(x0_human, pred_human)
    # OT between predicted human and real mouse brain — measures cross-species similarity
    ot_cross_pre = ot_distance(pred_human[:B], real_brain_mouse)
print(f"  Human input→output OT (pre-finetune): {ot_human_pre.item():.2f}")
print(f"  Human-predicted vs mouse-brain OT (pre-finetune): {ot_cross_pre.item():.2f}")

# ---- 7. Fine-tune on human data (lightweight) ----
print("\nFine-tuning on human data (20 epochs)...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=cfg['weight_decay'])
finetune_losses = []
for epoch in range(20):
    idx = np.random.choice(len(adni_tensor), min(B, len(adni_tensor)), replace=True)
    x0 = adni_tensor[idx]
    optimizer.zero_grad()
    pred = integrate_ode(model, x0, torch.tensor(0.0), torch.tensor(1.0), n_steps=4)
    # OT to mouse brain (cross-species alignment target)
    mouse_idx = np.random.choice(len(mouse_brain), min(B, len(mouse_brain)), replace=True)
    target = mouse_brain[mouse_idx]
    ot_loss = ot_distance(pred, target)
    # light L2 reg to prevent drift from pretrained weights (only matching param tensors)
    reg = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if name in ckpt['state_dict']:
            p0 = ckpt['state_dict'][name]
            if p.shape == p0.shape:
                reg = reg + (p - p0).pow(2).sum()
    reg = reg * 0.01
    total_loss = ot_loss + reg
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"  epoch {epoch+1}: NaN detected, skipping")
        continue
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    finetune_losses.append(total_loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}: loss={total_loss.item():.2f} (ot={ot_loss.item():.2f})")

# ---- 8. Re-evaluate after fine-tune ----
model.eval()
with torch.no_grad():
    pred_human_post = integrate_ode(model, x0_human, torch.tensor(0.0), torch.tensor(1.0), n_steps=4)
    ot_human_post = ot_distance(x0_human, pred_human_post)
    ot_cross_post = ot_distance(pred_human_post[:B], real_brain_mouse)

    # Re-evaluate mouse (should still work)
    pred_brain_mouse_post = integrate_ode(model, x0_mouse, torch.tensor(0.0), torch.tensor(1.0), n_steps=4)
    ot_mouse_post = ot_distance(pred_brain_mouse_post, real_brain_mouse)

print(f"\n=== Results ===")
print(f"  Mouse OT (pre → post):  {ot_mouse.item():.2f} → {ot_mouse_post.item():.2f}")
print(f"  Human I/O OT (pre → post): {ot_human_pre.item():.2f} → {ot_human_post.item():.2f}")
print(f"  Cross-species OT (pre → post): {ot_cross_pre.item():.2f} → {ot_cross_post.item():.2f}")

# ---- 9. Save results ----
results = {
    "mouse_ot_pre": float(ot_mouse.item()),
    "mouse_ot_post": float(ot_mouse_post.item()),
    "human_io_ot_pre": float(ot_human_pre.item()),
    "human_io_ot_post": float(ot_human_post.item()),
    "cross_species_ot_pre": float(ot_cross_pre.item()),
    "cross_species_ot_post": float(ot_cross_post.item()),
    "adni_subjects": int(adni_tensor.shape[0]),
    "genes_matched": len(adni_genes_in_model),
    "genes_total": len(common_genes),
    "finetune_epochs": 20,
    "finetune_final_loss": float(finetune_losses[-1]),
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_JSON}")
