import os
#!/usr/bin/env python3
"""
Fair read-only ablation evaluation for the existing Step1 transcriptomics model.

Principles:
1. Do NOT retrain or modify the original Step1 pipeline.
2. Load the existing Neural ODE checkpoint and original normalization parameters.
3. Use the original train/validation split seed and gene order.
4. Compare against lightweight baselines on the same held-out validation subset.
5. Write outputs only under output/ablation_existing_fair/.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config  # noqa: E402

config = get_config()
HEPAWORLD_DIR = str(config.get_path("paths.hepaworld_dir"))
sys.path.insert(0, HEPAWORLD_DIR)

from models.dynamics import DriftNet, integrate_ode  # noqa: E402
from utils.seed import set_global_seed  # noqa: E402

SEED = 42
VAL_FRAC = 0.1
MAX_EVAL = 1400
MAX_RIDGE_TRAIN = 3000
MAX_OT_ANCHORS = 512
BATCH_SIZE = 128
OT_EPSILON = 0.15
OT_ITERS = 50
MMD_GAMMA = 1.0 / 2000.0
DEVICE = "cpu"

DATA_DIR = Path(str(config.get_path("paths.processed_data_dir")))
BLOOD_PATH = DATA_DIR / "transcriptomics_blood.h5ad"
BRAIN_PATH = DATA_DIR / "transcriptomics_brain.h5ad"
STEP1_OUT = PROJECT_ROOT / "output/step1_world_model_transcriptomics_no_pca"
CKPT_PATH = STEP1_OUT / "checkpoints/best.pt"
NORM_PATH = STEP1_OUT / "normalization_params.npz"
OUT_DIR = PROJECT_ROOT / "output/ablation_existing_fair"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_backed_slice(xmat, row_idx, col_idx):
    """Read backed AnnData rows/columns as dense float32, preserving requested row order."""
    row_idx = np.asarray(row_idx)
    order = np.argsort(row_idx)
    sorted_rows = row_idx[order]
    block = xmat[sorted_rows, :]
    if sp.issparse(block):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float32)
    block = block[:, col_idx]
    restore = np.argsort(order)
    return block[restore]


def sinkhorn_cost(x, y, epsilon=OT_EPSILON, n_iters=OT_ITERS):
    """Entropic OT cost between two batches."""
    cost = torch.cdist(x, y, p=2.0) ** 2
    n, m = cost.shape
    mu = torch.full((n,), 1.0 / n, dtype=x.dtype, device=x.device)
    nu = torch.full((m,), 1.0 / m, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (kernel @ v + 1e-8)
        v = nu / (kernel.t() @ u + 1e-8)
    transport = torch.diag(u) @ kernel @ torch.diag(v)
    return torch.sum(transport * cost)


def sinkhorn_barycentric_map(x, anchors, epsilon=OT_EPSILON, n_iters=OT_ITERS):
    """Map x to target anchor space by barycentric projection using train-target anchors only."""
    cost = torch.cdist(x, anchors, p=2.0) ** 2
    n, m = cost.shape
    mu = torch.full((n,), 1.0 / n, dtype=x.dtype, device=x.device)
    nu = torch.full((m,), 1.0 / m, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (kernel @ v + 1e-8)
        v = nu / (kernel.t() @ u + 1e-8)
    transport = torch.diag(u) @ kernel @ torch.diag(v)
    row_mass = transport.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (transport / row_mass) @ anchors


def mean_batch_ot(pred, target, rng, n_batches=30):
    losses = []
    n = min(len(pred), len(target))
    for _ in range(n_batches):
        idx = rng.choice(n, min(BATCH_SIZE, n), replace=False)
        x = torch.from_numpy(pred[idx]).float()
        y = torch.from_numpy(target[idx]).float()
        losses.append(float(sinkhorn_cost(x, y).item()))
    return float(np.mean(losses)), float(np.std(losses))


def mmd_rbf(x, y, gamma=MMD_GAMMA, max_points=512, seed=SEED):
    rng = np.random.RandomState(seed)
    nx = min(max_points, len(x))
    ny = min(max_points, len(y))
    ix = rng.choice(len(x), nx, replace=False)
    iy = rng.choice(len(y), ny, replace=False)
    xt = torch.from_numpy(x[ix]).float()
    yt = torch.from_numpy(y[iy]).float()
    kxx = torch.exp(-gamma * torch.cdist(xt, xt, p=2.0) ** 2).mean()
    kyy = torch.exp(-gamma * torch.cdist(yt, yt, p=2.0) ** 2).mean()
    kxy = torch.exp(-gamma * torch.cdist(xt, yt, p=2.0) ** 2).mean()
    return float((kxx + kyy - 2.0 * kxy).item())


def safe_gene_corr(pred, target):
    vals = []
    for j in range(pred.shape[1]):
        a = pred[:, j]
        b = target[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    if not vals:
        return float("nan"), float("nan")
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def corr_structure_error(pred, target, n_genes=300, seed=SEED):
    rng = np.random.RandomState(seed)
    cols = rng.choice(pred.shape[1], min(n_genes, pred.shape[1]), replace=False)
    cp = np.corrcoef(pred[:, cols], rowvar=False)
    ct = np.corrcoef(target[:, cols], rowvar=False)
    cp = np.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)
    ct = np.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
    tri = np.triu_indices_from(cp, k=1)
    return float(np.mean(np.abs(cp[tri] - ct[tri])))


def evaluate_method(name, pred, target, rng):
    ot_mean, ot_sd = mean_batch_ot(pred, target, rng)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    gc_mean, gc_median = safe_gene_corr(pred, target)
    mmd = mmd_rbf(pred, target)
    cse = corr_structure_error(pred, target)
    return {
        "method": name,
        "ot_distance_mean": ot_mean,
        "ot_distance_sd": ot_sd,
        "mse_pseudo_aligned": mse,
        "mae_pseudo_aligned": mae,
        "gene_corr_mean": gc_mean,
        "gene_corr_median": gc_median,
        "mmd_rbf": mmd,
        "corr_structure_mae": cse,
    }


def main():
    t_start = time.time()
    set_global_seed(SEED, deterministic=True)
    rng = np.random.RandomState(SEED + 100)

    print("=" * 80)
    print("READ-ONLY FAIR ABLATION: existing Neural ODE vs baselines")
    print("=" * 80)
    print(f"[PATH] blood={BLOOD_PATH}")
    print(f"[PATH] brain={BRAIN_PATH}")
    print(f"[PATH] checkpoint={CKPT_PATH}")
    print(f"[PATH] normalization={NORM_PATH}")
    print(f"[OUT] {OUT_DIR}")

    if not BLOOD_PATH.exists() or not BRAIN_PATH.exists():
        raise FileNotFoundError("Missing processed transcriptomics h5ad files")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Missing normalization params: {NORM_PATH}")

    norm = np.load(NORM_PATH, allow_pickle=True)
    genes = [str(g) for g in norm["common_genes"]]
    blood_mean = norm["plasma_mean"].astype(np.float32)
    blood_std = norm["plasma_std"].astype(np.float32)
    brain_mean = norm["csf_mean"].astype(np.float32)
    brain_std = norm["csf_std"].astype(np.float32)
    print(f"[NORM] genes={len(genes)}")

    adata_blood = ad.read_h5ad(BLOOD_PATH, backed="r")
    adata_brain = ad.read_h5ad(BRAIN_PATH, backed="r")
    blood_var = list(map(str, adata_blood.var_names))
    brain_var = list(map(str, adata_brain.var_names))
    blood_lookup = {g: i for i, g in enumerate(blood_var)}
    brain_lookup = {g: i for i, g in enumerate(brain_var)}
    blood_cols = [blood_lookup[g] for g in genes]
    brain_cols = [brain_lookup[g] for g in genes]
    print(f"[DATA] blood={adata_blood.n_obs}x{adata_blood.n_vars}, brain={adata_brain.n_obs}x{adata_brain.n_vars}")

    split_rng = np.random.RandomState(SEED + 1)
    blood_perm = split_rng.permutation(adata_blood.n_obs)
    brain_perm = split_rng.permutation(adata_brain.n_obs)
    n_blood_train = int(adata_blood.n_obs * (1.0 - VAL_FRAC))
    n_brain_train = int(adata_brain.n_obs * (1.0 - VAL_FRAC))
    blood_train_idx = blood_perm[:n_blood_train]
    blood_val_idx = blood_perm[n_blood_train:]
    brain_train_idx = brain_perm[:n_brain_train]
    brain_val_idx = brain_perm[n_brain_train:]
    print(f"[SPLIT] train blood={len(blood_train_idx)}, brain={len(brain_train_idx)}")
    print(f"[SPLIT] val blood={len(blood_val_idx)}, brain={len(brain_val_idx)}")

    eval_n = min(MAX_EVAL, len(blood_val_idx), len(brain_val_idx))
    ridge_train_n = min(MAX_RIDGE_TRAIN, len(blood_train_idx), len(brain_train_idx))
    anchor_n = min(MAX_OT_ANCHORS, len(brain_train_idx))
    blood_eval_idx = rng.choice(blood_val_idx, eval_n, replace=False)
    brain_eval_idx = rng.choice(brain_val_idx, eval_n, replace=False)
    blood_ridge_idx = rng.choice(blood_train_idx, ridge_train_n, replace=False)
    brain_ridge_idx = rng.choice(brain_train_idx, ridge_train_n, replace=False)
    brain_anchor_idx = rng.choice(brain_train_idx, anchor_n, replace=False)
    print(f"[SUBSET] eval={eval_n}, ridge_train={ridge_train_n}, ot_anchors={anchor_n}")

    x_blood_eval = dense_backed_slice(adata_blood.X, blood_eval_idx, blood_cols)
    x_brain_eval = dense_backed_slice(adata_brain.X, brain_eval_idx, brain_cols)
    x_blood_eval = (x_blood_eval - blood_mean) / blood_std
    x_brain_eval = (x_brain_eval - brain_mean) / brain_std

    x_blood_train = dense_backed_slice(adata_blood.X, blood_ridge_idx, blood_cols)
    x_brain_train = dense_backed_slice(adata_brain.X, brain_ridge_idx, brain_cols)
    x_blood_train = (x_blood_train - blood_mean) / blood_std
    x_brain_train = (x_brain_train - brain_mean) / brain_std

    x_brain_anchor = dense_backed_slice(adata_brain.X, brain_anchor_idx, brain_cols)
    x_brain_anchor = (x_brain_anchor - brain_mean) / brain_std
    print("[LOAD] all evaluation arrays loaded and normalized")

    results = []

    print("\n[METHOD] Identity baseline")
    pred_identity = x_blood_eval.copy()
    results.append(evaluate_method("Identity", pred_identity, x_brain_eval, rng))
    print(json.dumps(results[-1], indent=2))

    print("\n[METHOD] Existing Neural ODE checkpoint")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = DriftNet(dim=len(genes), hidden=64, depth=2, time_freq=4, dropout=0.3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pred_ode = np.zeros_like(x_blood_eval, dtype=np.float32)
    t0 = torch.tensor(0.0)
    t1 = torch.tensor(1.0)
    with torch.no_grad():
        for start in range(0, eval_n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, eval_n)
            xb = torch.from_numpy(x_blood_eval[start:end]).float()
            yb = integrate_ode(lambda xx, tt: model(xx, tt), xb, t0, t1, 4, method="rk4")
            pred_ode[start:end] = yb.numpy()
    results.append(evaluate_method("NeuralODE_existing", pred_ode, x_brain_eval, rng))
    results[-1]["checkpoint_epoch"] = ckpt.get("epoch", "unknown")
    print(json.dumps(results[-1], indent=2))

    print("\n[METHOD] Ridge baseline on pseudo-aligned unpaired samples")
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(x_blood_train, x_brain_train)
    pred_ridge = ridge.predict(x_blood_eval).astype(np.float32)
    results.append(evaluate_method("Ridge_pseudo_aligned", pred_ridge, x_brain_eval, rng))
    print(json.dumps(results[-1], indent=2))

    print("\n[METHOD] Direct OT barycentric baseline using train-brain anchors only")
    anchors_t = torch.from_numpy(x_brain_anchor).float()
    pred_ot = np.zeros_like(x_blood_eval, dtype=np.float32)
    for start in range(0, eval_n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, eval_n)
        xb = torch.from_numpy(x_blood_eval[start:end]).float()
        mapped = sinkhorn_barycentric_map(xb, anchors_t)
        pred_ot[start:end] = mapped.numpy()
    results.append(evaluate_method("DirectOT_barycentric", pred_ot, x_brain_eval, rng))
    print(json.dumps(results[-1], indent=2))

    csv_path = OUT_DIR / "fair_ablation_results.csv"
    keys = list(results[0].keys())
    for row in results:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[SAVE] {csv_path}")

    summary = {
        "script": str(Path(__file__).resolve()),
        "read_only_inputs": {
            "blood_h5ad": str(BLOOD_PATH),
            "brain_h5ad": str(BRAIN_PATH),
            "checkpoint": str(CKPT_PATH),
            "normalization_params": str(NORM_PATH),
        },
        "outputs": {"csv": str(csv_path)},
        "parameters": {
            "seed": SEED,
            "val_frac": VAL_FRAC,
            "max_eval": MAX_EVAL,
            "max_ridge_train": MAX_RIDGE_TRAIN,
            "max_ot_anchors": MAX_OT_ANCHORS,
            "batch_size": BATCH_SIZE,
            "ot_epsilon": OT_EPSILON,
            "ot_iters": OT_ITERS,
        },
        "results": results,
        "elapsed_seconds": time.time() - t_start,
    }
    json_path = OUT_DIR / "fair_ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVE] {json_path}")

    methods = [r["method"] for r in results]
    metrics = ["ot_distance_mean", "mmd_rbf", "corr_structure_mae", "mse_pseudo_aligned"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    for ax, metric in zip(axes, metrics):
        vals = [r[metric] for r in results]
        ax.bar(methods, vals, color=["#888888", "#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_path = OUT_DIR / "fair_ablation_metrics.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {fig_path}")

    adata_blood.file.close()
    adata_brain.file.close()
    del pred_identity, pred_ode, pred_ridge, pred_ot
    gc.collect()
    print("\n✅ Fair ablation evaluation complete")


if __name__ == "__main__":
    main()
