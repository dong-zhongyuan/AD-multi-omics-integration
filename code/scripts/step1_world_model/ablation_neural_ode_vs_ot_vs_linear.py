import os
#!/usr/bin/env python3
"""Ablation: Neural ODE vs Direct OT Map vs Linear Mapping.
Evaluates blood→brain cross-tissue mapping quality.

Follows 8-file provenance: uses processed data from step1 output.
"""

import sys, os, json, time, gc
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))
from tools.config_loader import get_config
config = get_config()

Hepaworld_dir = str(config.get_path("paths.hepaworld_dir"))
sys.path.insert(0, Hepaworld_dir)

import anndata as ad
from models.dynamics import DriftNet, integrate_ode
from utils.seed import set_global_seed

# ─── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path(str(config.get_path("paths.processed_data_dir")))
BLOOD_PATH = DATA_DIR / "transcriptomics_blood.h5ad"
BRAIN_PATH = DATA_DIR / "transcriptomics_brain.h5ad"
OUT_DIR = PROJECT_ROOT / "output/ablation_neural_ode_vs_ot_vs_linear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cpu"
N_GENES = 2000
BATCH_SIZE = 128
N_BATCHES = 100  # number of validation batches for OT comparison
VAL_FRAC = 0.1
OT_EPSILON = 0.15
OT_ITERS = 50

set_global_seed(SEED, deterministic=True)

# ─── Sinkhorn OT Loss (same as step1) ──────────────────────────────────────
def sinkhorn_ot_loss(x, y, epsilon=OT_EPSILON, n_iters=OT_ITERS):
    C = torch.cdist(x, y, p=2.0) ** 2
    n, m = C.shape
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    mu = torch.full((n,), 1.0/n, device=x.device, dtype=x.dtype)
    nu = torch.full((m,), 1.0/m, device=x.device, dtype=x.dtype)
    K = torch.exp(-C / epsilon)
    K = torch.clamp(K, min=1e-8)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.t() @ u + 1e-8)
    T = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(T * C)

# ─── 1. Load Data ──────────────────────────────────────────────────────────
print("=" * 60)
print("ABLATION: Neural ODE vs Direct OT vs Linear Mapping")
print("=" * 60)
print(f"[LOAD] Loading backed h5ad files...")
adata_blood = ad.read_h5ad(BLOOD_PATH, backed='r')
adata_brain = ad.read_h5ad(BRAIN_PATH, backed='r')
print(f"[DATA] Blood: {adata_blood.n_obs} samples × {adata_blood.n_vars} genes")
print(f"[DATA] Brain: {adata_brain.n_obs} samples × {adata_brain.n_vars} genes")

# Common genes
common_genes = list(set(adata_blood.var_names) & set(adata_brain.var_names))
print(f"[DATA] Common genes: {len(common_genes)}")

blood_gene_idx = [list(adata_blood.var_names).index(g) for g in common_genes]
brain_gene_idx = [list(adata_brain.var_names).index(g) for g in common_genes]

def _dense_backed_slice(X, row_idx, col_idx):
    """Convert a backed h5ad row×col slice to a dense float32 numpy array."""
    block = X[row_idx, :][:, col_idx] if col_idx is not None else X[row_idx, :]
    if sp.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)

# HVG selection (top N_GENES by variance)
if len(common_genes) > N_GENES:
    print(f"[HVG] Selecting top {N_GENES} variable genes...")
    chunk_size = 1000
    
    def compute_var(adata, gene_indices):
        sum_x = np.zeros(len(gene_indices), dtype=np.float32)
        sum_sq = np.zeros(len(gene_indices), dtype=np.float32)
        count = 0
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            chunk = _dense_backed_slice(adata.X, slice(start, end), gene_indices)
            sum_x += chunk.sum(axis=0)
            sum_sq += (chunk ** 2).sum(axis=0)
            count += chunk.shape[0]
            del chunk; gc.collect()
        mean = sum_x / count
        var = (sum_sq / count) - mean**2
        return var
    
    var_blood = compute_var(adata_blood, blood_gene_idx)
    var_brain = compute_var(adata_brain, brain_gene_idx)
    var_combined = var_blood + var_brain
    top_idx = np.argsort(var_combined)[-N_GENES:]
    
    common_genes = [common_genes[i] for i in top_idx]
    blood_gene_idx = [blood_gene_idx[i] for i in top_idx]
    brain_gene_idx = [brain_gene_idx[i] for i in top_idx]
    print(f"[HVG] Selected {len(common_genes)} genes")

# ─── 2. Normalize ──────────────────────────────────────────────────────────
print("\n[PREP] Computing normalization stats...")

def compute_mean_std(adata, gene_indices):
    chunk_size = 1000
    n = len(gene_indices)
    sum_x = np.zeros(n, dtype=np.float64)
    sum_sq = np.zeros(n, dtype=np.float64)
    count = 0
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)
        chunk = _dense_backed_slice(adata.X, slice(start, end), gene_indices)
        sum_x += chunk.sum(axis=0).astype(np.float64)
        sum_sq += (chunk.astype(np.float64) ** 2).sum(axis=0)
        count += chunk.shape[0]
        del chunk; gc.collect()
    mean = (sum_x / count).astype(np.float32)
    std = (np.sqrt(sum_sq / count - mean**2) + 1e-8).astype(np.float32)
    return mean, std

blood_mean, blood_std = compute_mean_std(adata_blood, blood_gene_idx)
brain_mean, brain_std = compute_mean_std(adata_brain, brain_gene_idx)
print(f"[PREP] Blood mean range: [{blood_mean.min():.3f}, {blood_mean.max():.3f}]")
print(f"[PREP] Brain mean range: [{brain_mean.min():.3f}, {brain_mean.max():.3f}]")

# ─── 3. Train/Val Split ────────────────────────────────────────────────────
print("\n[SPLIT] Creating train/val split...")
rng = np.random.RandomState(SEED + 1)
blood_idx = rng.permutation(adata_blood.n_obs)
brain_idx = rng.permutation(adata_brain.n_obs)

n_blood_train = int(adata_blood.n_obs * (1 - VAL_FRAC))
n_brain_train = int(adata_brain.n_obs * (1 - VAL_FRAC))

blood_train = blood_idx[:n_blood_train]
blood_val = blood_idx[n_blood_train:]
brain_train = brain_idx[:n_brain_train]
brain_val = brain_idx[n_brain_train:]
print(f"[SPLIT] Blood: train={len(blood_train)}, val={len(blood_val)}")
print(f"[SPLIT] Brain: train={len(brain_train)}, val={len(brain_val)}")

# ─── Load all training data into memory (for linear model) ─────────────────
print("\n[LOAD] Loading training data into memory...")

def load_all(adata, indices, gene_idx, mean, std):
    # h5py requires sorted indices
    sort_order = np.argsort(indices)
    idx_sorted = indices[sort_order]
    X = _dense_backed_slice(adata.X, idx_sorted, gene_idx)
    X = X[np.argsort(sort_order)]  # restore order
    X_norm = (X - mean) / std
    del X; gc.collect()
    return X_norm

X_blood_train = load_all(adata_blood, blood_train, blood_gene_idx, blood_mean, blood_std)
X_brain_train = load_all(adata_brain, brain_train, brain_gene_idx, brain_mean, brain_std)
print(f"[LOAD] Train: blood {X_blood_train.shape}, brain {X_brain_train.shape}")

# Align sample counts for supervised methods (use smaller cohort size)
rng_align = np.random.RandomState(SEED + 2)
n_train_align = min(len(X_blood_train), len(X_brain_train))
blood_align_idx = rng_align.choice(len(X_blood_train), n_train_align, replace=False)
brain_align_idx = rng_align.choice(len(X_brain_train), n_train_align, replace=False)
X_blood_train_align = X_blood_train[blood_align_idx]
X_brain_train_align = X_brain_train[brain_align_idx]
print(f"[ALIGN] Both train sets: {n_train_align} samples")

# ─── Load validation data ──────────────────────────────────────────────────
X_blood_val = load_all(adata_blood, blood_val, blood_gene_idx, blood_mean, blood_std)
X_brain_val = load_all(adata_brain, brain_val, brain_gene_idx, brain_mean, brain_std)
print(f"[LOAD] Val: blood {X_blood_val.shape}, brain {X_brain_val.shape}")

# Align validation
n_val_align = min(len(X_blood_val), len(X_brain_val))
blood_val_align_idx = rng_align.choice(len(X_blood_val), n_val_align, replace=False)
brain_val_align_idx = rng_align.choice(len(X_brain_val), n_val_align, replace=False)
X_blood_val_align = X_blood_val[blood_val_align_idx]
X_brain_val_align = X_brain_val[brain_val_align_idx]
print(f"[ALIGN] Both val sets: {n_val_align} samples")

# ─── 4. Method 3: Linear Mapping (Ridge Regression) ────────────────────────
print("\n" + "=" * 60)
print("METHOD 3: Linear Mapping (Ridge Regression)")
print("=" * 60)
t0 = time.time()
ridge = Ridge(alpha=1.0, fit_intercept=False, solver='auto')
ridge.fit(X_blood_train_align, X_brain_train_align)
X_brain_pred_linear = ridge.predict(X_blood_val_align)
linear_time = time.time() - t0

# MSE
linear_mse = np.mean((X_brain_pred_linear - X_brain_val_align) ** 2)
# OT loss on batches
linear_ot_losses = []
for _ in range(N_BATCHES):
    b = np.random.choice(len(X_brain_val_align), BATCH_SIZE, replace=False)
    pred_t = torch.from_numpy(X_brain_pred_linear[b]).float()
    real_t = torch.from_numpy(X_brain_val_align[b]).float()
    ot = float(sinkhorn_ot_loss(pred_t, real_t).item())
    linear_ot_losses.append(ot)
linear_ot = np.mean(linear_ot_losses)
# Gene-wise correlation
linear_corr = np.mean([np.corrcoef(X_brain_pred_linear[:,i], X_brain_val_align[:,i])[0,1]
                       for i in range(X_brain_val_align.shape[1])])
print(f"  Training time: {linear_time:.2f}s")
print(f"  Val MSE: {linear_mse:.6f}")
print(f"  Val OT loss (mean of {N_BATCHES} batches): {linear_ot:.4f}")
print(f"  Mean gene-wise Pearson r: {linear_corr:.4f}")

# ─── 5. Method 2: Direct OT Map (No Model) ─────────────────────────────────
print("\n" + "=" * 60)
print("METHOD 2: Direct OT Map (Sinkhorn coupling only)")
print("=" * 60)
t0 = time.time()
# Compute OT coupling between val blood and val brain
Xb_t = torch.from_numpy(X_blood_val).float()
Xc_t = torch.from_numpy(X_brain_val).float()

# For direct OT loss, compare raw distributions
direct_ot_losses = []
for _ in range(N_BATCHES):
    b = np.random.choice(len(X_blood_val), BATCH_SIZE, replace=False)
    c = np.random.choice(len(X_brain_val), BATCH_SIZE, replace=False)
    pred_t = Xb_t[b]
    real_t = Xc_t[c]
    ot = float(sinkhorn_ot_loss(pred_t, real_t).item())
    direct_ot_losses.append(ot)
direct_ot = np.mean(direct_ot_losses)
direct_time = time.time() - t0
# For correlation, use aligned validation sets
direct_corr = np.mean([np.corrcoef(X_blood_val_align[:,i], X_brain_val_align[:,i])[0,1]
                        for i in range(X_brain_val_align.shape[1])])
print(f"  Time: {direct_time:.2f}s")
print(f"  Val OT loss (mean of {N_BATCHES} batches): {direct_ot:.4f}")
print(f"  Mean gene-wise Pearson r (raw): {direct_corr:.4f}")

# ─── 6. Method 1: Neural ODE (load pre-trained model) ──────────────────────
print("\n" + "=" * 60)
print("METHOD 1: Neural ODE (DriftNet + RK4 integration)")
print("=" * 60)

# Load best checkpoint
ckpt_dir = PROJECT_ROOT / "output/step1_world_model_transcriptomics_no_pca"
ckpt_path = ckpt_dir / "checkpoints/best.pt"

if not ckpt_path.exists():
    print(f"[SKIP] No checkpoint at {ckpt_path}, cannot evaluate Neural ODE")
    neural_ot = None
    neural_corr = None
else:
    gene_dim = len(common_genes)
    model = DriftNet(dim=gene_dim, hidden=64, depth=2, time_freq=4, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"[LOAD] Checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

    # Apply Neural ODE to validation blood samples
    t0 = time.time()
    X_blood_val_t = torch.from_numpy(X_blood_val_align).float()
    t_start = torch.tensor(0.0).float()
    t_end = torch.tensor(1.0).float()

    # Process in batches due to memory
    n_val = len(X_blood_val_align)
    X_brain_pred_ode = np.zeros_like(X_blood_val_align)
    batch_size_ode = 200
    for start in range(0, n_val, batch_size_ode):
        end = min(start + batch_size_ode, n_val)
        xb = X_blood_val_t[start:end]
        with torch.no_grad():
            y = integrate_ode(
                lambda xx, tt: model(xx, tt),
                xb, t_start, t_end, n_steps=4, method='rk4'
            )
        X_brain_pred_ode[start:end] = y.cpu().numpy()
    
    neural_time = time.time() - t0
    
    # OT loss on batches
    neural_ot_losses = []
    for _ in range(N_BATCHES):
        b = np.random.choice(n_val, BATCH_SIZE, replace=False)
        pred_t = torch.from_numpy(X_brain_pred_ode[b]).float()
        real_t = torch.from_numpy(X_brain_val_align[b]).float()
        ot = float(sinkhorn_ot_loss(pred_t, real_t).item())
        neural_ot_losses.append(ot)
    neural_ot = np.mean(neural_ot_losses)
    neural_corr = np.mean([np.corrcoef(X_brain_pred_ode[:,i], X_brain_val_align[:,i])[0,1]
                           for i in range(X_brain_val_align.shape[1])])
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Inference time ({n_val} samples): {neural_time:.2f}s")
    print(f"  Val OT loss (mean of {N_BATCHES} batches): {neural_ot:.4f}")
    print(f"  Mean gene-wise Pearson r: {neural_corr:.4f}")

# ─── 7. Summary and Save ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLATION RESULTS SUMMARY")
print("=" * 60)

results = {
    'method': ['Neural ODE', 'Direct OT', 'Linear (Ridge)'],
    'ot_loss': [neural_ot if neural_ot else float('nan'), direct_ot, linear_ot],
    'gene_corr': [neural_corr if neural_corr else float('nan'), direct_corr, linear_corr],
}

print(f"\n{'Method':<20} {'OT Loss ↓':<15} {'Gene Corr r ↑':<15}")
print("-" * 50)
for i in range(3):
    ot_str = f"{results['ot_loss'][i]:.4f}" if not np.isnan(results['ot_loss'][i]) else "N/A"
    corr_str = f"{results['gene_corr'][i]:.4f}" if not np.isnan(results['gene_corr'][i]) else "N/A"
    print(f"{results['method'][i]:<20} {ot_str:<15} {corr_str:<15}")

# Save results
import csv
csv_path = OUT_DIR / 'ablation_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(results.keys())
    for i in range(3):
        w.writerow([results[k][i] for k in results.keys()])
print(f"\nResults saved: {csv_path}")

# Close backed files
adata_blood.file.close()
adata_brain.file.close()

print("\n✅ Ablation complete!")
