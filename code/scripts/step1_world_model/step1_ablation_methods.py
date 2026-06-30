import os
#!/usr/bin/env python3
"""Ablation: Neural ODE vs Direct OT vs Linear Mapping.
Evaluates three methods for blood→brain cross-tissue mapping on the same validation data.
Output goes to output/ablation_method_comparison/ — parallel to other step outputs.
"""
from __future__ import annotations

import gc, json, sys, time
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))
from tools.config_loader import get_config
config = get_config()

HEPAWORLD_DIR = str(config.get_path("paths.hepaworld_dir"))
sys.path.insert(0, HEPAWORLD_DIR)
from models.dynamics import DriftNet, integrate_ode
from utils.seed import set_global_seed

OUT_DIR = PROJECT_ROOT / "output/ablation_method_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42; DEVICE = "cpu"
N_GENES = 2000; BATCH_SIZE = 128
VAL_FRAC = 0.1; OT_EPSILON = 0.15; OT_ITERS = 50

set_global_seed(SEED, deterministic=True)

# ─── Sinkhorn OT Loss ────────────────────────────────────────────────────
def sinkhorn_ot_loss(x, y, epsilon=0.15, n_iters=50):
    C = torch.cdist(x, y, p=2.0) ** 2
    n, m = C.shape
    if n == 0 or m == 0:
        return torch.tensor(0.0)
    mu = torch.full((n,), 1.0/n); nu = torch.full((m,), 1.0/m)
    K = torch.exp(-C / epsilon); K = torch.clamp(K, min=1e-8)
    u = torch.ones_like(mu); v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (K @ v + 1e-8); v = nu / (K.t() @ u + 1e-8)
    T = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(T * C)

def _dense_slice(X, row_idx):
    block = X[row_idx, :]
    if sp.issparse(block): block = block.toarray()
    return np.asarray(block, dtype=np.float32)

# ─── 1. Load Data ───────────────────────────────────────────────────────
print("=" * 60)
print("ABLATION: Neural ODE vs Direct OT vs Linear Mapping")
print("=" * 60)

DATA_DIR = Path(str(config.get_path("paths.processed_data_dir")))
blood_path = DATA_DIR / "transcriptomics_blood.h5ad"
brain_path = DATA_DIR / "transcriptomics_brain.h5ad"

ada_blood = ad.read_h5ad(blood_path, backed='r')
ada_brain = ad.read_h5ad(brain_path, backed='r')
print(f"[DATA] Blood: {ada_blood.n_obs}×{ada_blood.n_vars}, Brain: {ada_brain.n_obs}×{ada_brain.n_vars}")

common = list(set(ada_blood.var_names) & set(ada_brain.var_names))
print(f"[DATA] Common genes: {len(common)}")

b_gidx = [list(ada_blood.var_names).index(g) for g in common]
n_gidx = [list(ada_brain.var_names).index(g) for g in common]

# HVG selection
if len(common) > N_GENES:
    # Chunked variance
    cs = 500
    n_genes = len(common)
    s1 = np.zeros(n_genes, dtype=np.float64); s2 = np.zeros(n_genes, dtype=np.float64)
    c1 = 0
    for st in range(0, ada_blood.n_obs, cs):
        en = min(st+cs, ada_blood.n_obs)
        ch = _dense_slice(ada_blood.X, slice(st,en))[:, b_gidx]
        s1 += ch.sum(axis=0); s2 += (ch**2).sum(axis=0); c1 += ch.shape[0]; del ch
    v1 = s2/c1 - (s1/c1)**2

    s1f = np.zeros(n_genes, dtype=np.float64); s2f = np.zeros(n_genes, dtype=np.float64)
    c2 = 0
    for st in range(0, ada_brain.n_obs, cs):
        en = min(st+cs, ada_brain.n_obs)
        ch = _dense_slice(ada_brain.X, slice(st,en))[:, n_gidx]
        s1f += ch.sum(axis=0); s2f += (ch**2).sum(axis=0); c2 += ch.shape[0]; del ch
    v2 = s2f/c2 - (s1f/c2)**2

    top = np.argsort(v1+v2)[-N_GENES:]
    common = [common[i] for i in top]; b_gidx = [b_gidx[i] for i in top]; n_gidx = [n_gidx[i] for i in top]
    print(f"[DATA] HVG filtered: {len(common)}")

# Train/val split
rng = np.random.RandomState(SEED+1)
n_blood, n_brain = ada_blood.n_obs, ada_brain.n_obs
plasma_idx = rng.permutation(n_blood)
csf_idx = rng.permutation(n_brain)
n_blood_train = int(n_blood*(1-VAL_FRAC)); n_brain_train = int(n_brain*(1-VAL_FRAC))
b_train_idx = plasma_idx[:n_blood_train]; b_val_idx = plasma_idx[n_blood_train:]
n_train_idx = csf_idx[:n_brain_train]; n_val_idx = csf_idx[n_brain_train:]
print(f"[SPLIT] Train blood={len(b_train_idx)} brain={n_brain_train}, Val blood={len(b_val_idx)} brain={len(n_val_idx)}")

# ─── 2. Load training and validation data ────────────────
print("[LOAD] Loading traing data...")

def load_samples(adata, idx, gene_idx):
    """Load samples ensuring sorted access for backed h5ad."""
    order = np.argsort(idx)
    sorted_idx = idx[order]
    X = _dense_slice(adata.X, sorted_idx)[:, gene_idx]
    # Restore original order
    X = X[np.argsort(order)]
    return X

X_blood_train = load_samples(ada_blood, b_train_idx, b_gidx)
X_brain_train = load_samples(ada_brain, n_train_idx, n_gidx)
print(f"[TRAIN] Blood: {X_blood_train.shape}, Brain: {X_brain_train.shape}")

print("[LOAD] Loading validation data...")
X_blood_val = load_samples(ada_blood, b_val_idx, b_gidx)
X_brain_val = load_samples(ada_brain, n_val_idx, n_gidx)
print(f"[VAL] Blood: {X_blood_val.shape}, Brain: {X_brain_val.shape}")

# Compute normalization stats from training data only
b_mean = X_blood_train.mean(axis=0); b_std = X_blood_train.std(axis=0) + 1e-8
n_mean = X_brain_train.mean(axis=0); n_std = X_brain_train.std(axis=0) + 1e-8

# Normalize both train and val using train stats
X_blood_train_norm = (X_blood_train - b_mean) / b_std
X_brain_train_norm = (X_brain_train - n_mean) / n_std
X_blood_val_norm = (X_blood_val - b_mean) / b_std
X_brain_val_norm = (X_brain_val - n_mean) / n_std

print(f"[NORM] Train: blood {X_blood_train_norm.shape}, brain {X_brain_train_norm.shape}")
print(f"[NORM] Val: blood {X_blood_val_norm.shape}, brain {X_brain_val_norm.shape}")

# Align train/val sample counts for methods that need pairing
n_train_align = min(len(X_blood_train), len(X_brain_train))
n_val_align = min(len(X_blood_val), len(X_brain_val))

rng_align = np.random.RandomState(SEED+2)
train_blood_idx = rng_align.choice(len(X_blood_train), n_train_align, replace=False)
train_brain_idx = rng_align.choice(len(X_brain_train), n_train_align, replace=False)
val_blood_idx = rng_align.choice(len(X_blood_val), n_val_align, replace=False)
val_brain_idx = rng_align.choice(len(X_brain_val), n_val_align, replace=False)

X_blood_train_align = X_blood_train_norm[train_blood_idx]
X_brain_train_align = X_brain_train_norm[train_brain_idx]
X_blood_val_align = X_blood_val_norm[val_blood_idx]
X_brain_val_align = X_brain_val_norm[val_brain_idx]

print(f"[ALIGN] Train: {n_train_align} samples, Val: {n_val_align} samples")

# ─── 3. Evaluate Three Methods ───────────────────────────────
results = {}

# ── 3a. BASELINE: Identity (no mapping) ──
print("\n[METRIC 0] Baseline: Identity mapping (blood→blood vs brain)")
X_base_t = torch.tensor(X_blood_val_align, dtype=torch.float32)
X_target_t = torch.tensor(X_brain_val_align, dtype=torch.float32)
base_ot = float(sinkhorn_ot_loss(X_base_t, X_target_t, OT_EPSILON, OT_ITERS))
base_mse = float(np.mean((X_blood_val_align - X_brain_val_align)**2))
results['Identity'] = {'ot_distance': base_ot, 'mse': base_mse}
print(f"  OT={base_ot:.4f}, MSE={base_mse:.6f}")

# ── 3b. NEURAL ODE (load trained checkpoint) ──
print("\n[METRIC 1] Neural ODE (DriftNet + RK4)")

ckpt_dir = PROJECT_ROOT / "output/step1_world_model_transcriptomics_no_pca/checkpoints"
if not ckpt_dir.exists():
    # Try alternative path
    alt = list(PROJECT_ROOT.glob("output/step1*/checkpoints/best.pt"))
    if alt:
        best_path = alt[0]
    else:
        print("[SKIP] No Neural ODE checkpoint found!")
        results['NeuralODE'] = {'ot_distance': float('nan'), 'mse': float('nan')}
else:
    best_path = ckpt_dir / "best.pt"

if 'best_path' in dir() and best_path.exists():
    ckpt = torch.load(best_path, map_location='cpu')
    gene_dim = ckpt['cfg']['gene_dim']
    model = DriftNet(dim=gene_dim, hidden=64, depth=2, time_freq=4, dropout=0.3)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

  # Map blood → brain via ODE
    X_blood_t = torch.tensor(X_blood_val_align, dtype=torch.float32)
    X_brain_t = torch.tensor(X_brain_val_align, dtype=torch.float32)
    t0 = torch.tensor(0.0); t1 = torch.tensor(1.0)
    with torch.no_grad():
        y_ode = integrate_ode(lambda xx, tt: model(xx, tt), X_blood_t, t0, t1, 4, method='rk4')
        ode_ot = float(sinkhorn_ot_loss(y_ode, X_brain_t, OT_EPSILON, OT_ITERS))
        ode_mse = float(torch.mean((y_ode - X_brain_t)**2))
    results['NeuralODE'] = {'ot_distance': ode_ot, 'mse': ode_mse}
    print(f"  OT={ode_ot:.4f}, MSE={ode_mse:.6f}")
else:
    print("[SKIP] No checkpoint found, skipping Neural ODE")
    results['NeuralODE'] = {'ot_distance': float('nan'), 'mse': float('nan')}

# ── 3c. DIRECT OT MAP ──
print("\n[METRIC 2] Direct Sinkhorn OT Map")
print("  Training transport plan on training data...")
# Compute Sinkhorn transport plan on TRAINING data
X_b_train_t = torch.tensor(X_blood_train_align, dtype=torch.float32)
X_n_train_t = torch.tensor(X_brain_train_align, dtype=torch.float32)

C_train = torch.cdist(X_b_train_t, X_n_train_t, p=2.0) ** 2
mu_train = torch.full((n_train_align,), 1.0/n_train_align)
nu_train = torch.full((n_train_align,), 1.0/n_train_align)
K_train = torch.exp(-C_train / OT_EPSILON); K_train = torch.clamp(K_train, min=1e-8)
u_train = torch.ones_like(mu_train); v_train = torch.ones_like(nu_train)
for _ in range(OT_ITERS):
    u_train = mu_train / (K_train @ v_train + 1e-8)
    v_train = nu_train / (K_train.t() @ u_train + 1e-8)
T_train = torch.diag(u_train) @ K_train @ torch.diag(v_train)

# Apply to VALIDATION data: compute transport from val_blood to train_brain
print("  Applying to validation data...")
X_b_val_t = torch.tensor(X_blood_val_align, dtype=torch.float32)
X_n_val_t = torch.tensor(X_brain_val_align, dtype=torch.float32)

# Use the learned transport to map val blood → brain
# Approximate: use train brain as reference, compute new coupling
C_val = torch.cdist(X_b_val_t, X_n_train_t, p=2.0) ** 2
mu_val = torch.full((n_val_align,), 1.0/n_val_align)
nu_val = torch.full((n_train_align,), 1.0/n_train_align)
K_val = torch.exp(-C_val / OT_EPSILON); K_val = torch.clamp(K_val, min=1e-8)
u_val = torch.ones_like(mu_val); v_val = torch.ones_like(nu_val)
for _ in range(OT_ITERS):
    u_val = mu_val / (K_val @ v_val + 1e-8)
    v_val = nu_val / (K_val.t() @ u_val + 1e-8)
T_val = torch.diag(u_val) @ K_val @ torch.diag(v_val)

# Barycentric projection
T_val_np = T_val.numpy()
X_ot_mapped = n_train_align * T_val_np @ X_brain_train_align  # map to train brain space
X_ot_t = torch.tensor(X_ot_mapped, dtype=torch.float32)
ot_ot = float(sinkhorn_ot_loss(X_ot_t, X_n_val_t, OT_EPSILON, OT_ITERS))
ot_mse = float(np.mean((X_ot_mapped - X_brain_val_align)**2))
results['DirectOT'] = {'ot_distance': ot_ot, 'mse': ot_mse}
print(f"  OT={ot_ot:.4f}, MSE={ot_mse:.6f}")

# ── 3d. LINEAR MAPPING (Ridge) ──
print("\n[METRIC 3] Linear Ridge Regression")
print("  Training on training data...")
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_blood_train_align, X_brain_train_align)
print("  Predicting on validation data...")
X_ridge = ridge.predict(X_blood_val_align)
X_ridge_t = torch.tensor(X_ridge, dtype=torch.float32)
X_n_val_t = torch.tensor(X_brain_val_align, dtype=torch.float32)
ridge_ot = float(sinkhorn_ot_loss(X_ridge_t, X_n_val_t, OT_EPSILON, OT_ITERS))
ridge_mse = float(np.mean((X_ridge - X_brain_val_align)**2))
results['Linear'] = {'ot_distance': ridge_ot, 'mse': ridge_mse}
print(f"  OT={ridge_ot:.4f}, MSE={ridge_mse:.6f}")

# ── 3e. Per-gene correlation ──
print("\n[PER-GENE] Computing per-gene correlations...")
from scipy.stats import pearsonr

def per_gene_corr(X_pred, X_true):
    corrs = []
    for j in range(X_pred.shape[1]):
        if np.std(X_pred[:, j]) > 1e-8 and np.std(X_true[:, j]) > 1e-8:
            r, _ = pearsonr(X_pred[:, j], X_true[:, j])
            corrs.append(r)
        else:
            corrs.append(0.0)
    return np.array(corrs)

# Compute for each method that produced predictions
if results.get('NeuralODE', {}).get('mse', float('nan')) == float('nan'):
    ode_corrs = None
else:
    y_ode_np = y_ode.numpy()
    ode_corrs = per_gene_corr(y_ode_np, X_brain_val_align)

ot_corrs = per_gene_corr(X_ot_mapped, X_brain_val_align)
ridge_corrs = per_gene_corr(X_ridge, X_brain_val_align)

# ─── 4. Save Results ───────────────────────────────────────────────────
print("\n[SAVE] Writing results...")

# Summary JSON
summary = {}
for method, m in results.items():
    summary[method] = {'ot_distance': m['ot_distance'], 'mse': m['mse']}

# Add per-gene stats
summary['per_gene_correlation'] = {
    'DirectOT_mean': float(np.mean(ot_corrs)),
    'DirectOT_median': float(np.median(ot_corrs)),
    'Linear_mean': float(np.mean(ridge_corrs)),
    'Linear_median': float(np.median(ridge_corrs)),
}
if ode_corrs is not None:
    summary['per_gene_correlation']['NeuralODE_mean'] = float(np.mean(ode_corrs))
    summary['per_gene_correlation']['NeuralODE_median'] = float(np.median(ode_corrs))

with (OUT_DIR / 'summary.json').open('w') as f:
    json.dump(summary, f, indent=2)

# Per-gene CSV
import pandas as pd
gene_df = {'gene': common[:N_GENES], 'ot_corr': ot_corrs, 'ridge_corr': ridge_corrs}
if ode_corrs is not None:
    gene_df['ode_corr'] = ode_corrs
pd.DataFrame(gene_df).to_csv(OUT_DIR / 'per_gene_correlations.csv', index=False)

# ─── 5. Print Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLATION RESULTS")
print("=" * 60)
print(f"{'Method':<15} {'OT Distance':<15} {'MSE':<15} {'Gene Corr (mean)':<18}")
print("-" * 60)
for method in ['Identity', 'NeuralODE', 'DirectOT', 'Linear']:
    m = results.get(method, {})
    ot = m.get('ot_distance', float('nan'))
    mse = m.get('mse', float('nan'))
    corr_key = f'{method}_mean' if method != 'Identity' else None
    corr = summary.get('per_gene_correlation', {}).get(corr_key, '—') if corr_key else '—'
    print(f"{method:<15} {ot:<15.4f} {mse:<15.6f} {str(corr):<18}")

print(f"\nResults saved to: {OUT_DIR}")
