#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step4 - Dynamics training with Neural ODE + OT (engineered version)
===================================================================

保持：
- 默认路径与超参不变；
- 输出目录仍为 output/step4_dynamics_ot_ode；
- 产物文件名不变（preproc_latent.npz / checkpoints/best.pt / vectorfield_samples.best.npz）。

新增：
- Step4Config + JSON/YAML 配置；
- deterministic 模式；
- 代码拆分到 hepaworld.models / hepaworld.data 里。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import scanpy as sc
import torch

from ..config import Step4Config, load_step4_config
from ..data.preprocess import (
    build_time_bins,
    preprocess_hvg_pca,
    split_bins,
)
from ..models.dynamics import DriftNet, integrate_ode
from ..utils.logging import log, write_manifest_step4
from ..utils.seed import set_global_seed


def sinkhorn_ot_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    """Simple entropic OT loss between two batches x and y."""
    C = torch.cdist(x, y, p=2.0) ** 2  # (B, B)
    n, m = C.shape
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    mu = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    nu = torch.full((m,), 1.0 / m, device=x.device, dtype=x.dtype)

    K = torch.exp(-C / epsilon)
    eps = 1e-8
    K = torch.clamp(K, min=eps)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(n_iters):
        u = mu / (K @ v + eps)
        v = nu / (K.t() @ u + eps)

    T = torch.diag(u) @ K @ torch.diag(v)
    loss = torch.sum(T * C)
    return loss


def evaluate_epoch(
    model: DriftNet,
    X_latent: np.ndarray,
    times_sorted: List[float],
    val_bins: Dict[float, np.ndarray],
    cfg: Step4Config,
) -> Dict[str, float]:
    """Compute validation OT + drift regularisation over all adjacent time pairs."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.eval()
    pairs = list(zip(times_sorted[:-1], times_sorted[1:]))

    ot_losses: List[float] = []
    reg_losses: List[float] = []

    with torch.no_grad():
        for t0, t1 in pairs:
            idx0 = val_bins.get(float(t0))
            idx1 = val_bins.get(float(t1))
            if idx0 is None or idx1 is None or len(idx0) == 0 or len(idx1) == 0:
                continue
            B = min(cfg.batch_size, len(idx0), len(idx1))
            if B <= 1:
                continue
            sel0 = np.random.choice(idx0, size=B, replace=False)
            sel1 = np.random.choice(idx1, size=B, replace=False)

            x0 = torch.from_numpy(X_latent[sel0]).to(device)
            x1 = torch.from_numpy(X_latent[sel1]).to(device)

            t0_t = torch.tensor(float(t0), dtype=torch.float32, device=device)
            t1_t = torch.tensor(float(t1), dtype=torch.float32, device=device)

            y_hat = integrate_ode(
                lambda xx, tt: model(xx, tt),
                x0,
                t0_t,
                t1_t,
                cfg.n_steps_per_interval,
                method=cfg.integrator,
            )

            ot = sinkhorn_ot_loss(y_hat, x1, cfg.ot_epsilon, cfg.sinkhorn_iters)

            t_mid = 0.5 * (t0_t + t1_t)
            f_mid = model(x0, t_mid)
            reg = (f_mid.pow(2).sum(dim=1)).mean()

            ot_losses.append(float(ot.item()))
            reg_losses.append(float(reg.item()))

    if not ot_losses:
        return {"val_total": float("inf"), "val_ot": float("inf"), "val_reg": 0.0}

    val_ot = float(np.mean(ot_losses))
    val_reg = float(np.mean(reg_losses))
    val_total = val_ot + cfg.loss_reg_drift * val_reg
    return {"val_total": val_total, "val_ot": val_ot, "val_reg": val_reg}


def train_step4(cfg: Step4Config) -> None:
    """Main training entry for Step4 dynamics."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Save config & manifest early
    with (out_dir / "config_step4.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    write_manifest_step4(cfg, out_dir)

    # Seeds
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log(f"[INIT] device={device}, in_h5ad={cfg.in_h5ad}")
    log(f"[INIT] out_dir={out_dir}")

    # Load AnnData
    log("[DATA] reading AnnData ...")
    adata = sc.read_h5ad(cfg.in_h5ad)
    log(f"[DATA] shape cells={adata.n_obs:,} genes={adata.n_vars:,}")

    # Preprocess -> latent
    X_latent, t, times_sorted, hvg_genes = preprocess_hvg_pca(
        adata,
        n_top_hvg=cfg.n_top_hvg,
        pca_dim=cfg.pca_dim,
        normalize_target_sum=cfg.normalize_target_sum,
        out_dir=out_dir,
        batch_key=cfg.hvg_batch_key,
        time_key=cfg.time_key,
        random_state=cfg.seed,
    )

    # Build time bins & splits (train/val)
    log("[SPLIT] building time bins ...")
    bins_all = build_time_bins(t, times_sorted)
    train_bins, val_bins = split_bins(bins_all, cfg.val_frac, seed=cfg.seed + 1)

    for tt in times_sorted:
        n_all = len(bins_all[tt])
        n_tr = len(train_bins[tt])
        n_val = len(val_bins[tt])
        log(f"[SPLIT] time={tt}: total={n_all}, train={n_tr}, val={n_val}")

    # Move latent to numpy float32
    X_latent = np.asarray(X_latent, dtype=np.float32)

    # Build model & optimizer
    model = DriftNet(dim=cfg.pca_dim, hidden=256, depth=3, time_freq=4, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training bookkeeping
    pairs = list(zip(times_sorted[:-1], times_sorted[1:]))
    if not pairs:
        raise RuntimeError("Need at least two distinct time points for dynamics training.")

    best_val = float("inf")
    best_epoch = -1
    patience = 10
    no_improve = 0

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_total,train_ot,train_reg,val_total,val_ot,val_reg,sec\n")

    # Main epoch loop
    for epoch in range(1, cfg.epochs + 1):
        t_start = time.time()
        model.train()
        train_ot_losses: List[float] = []
        train_reg_losses: List[float] = []

        for (t0, t1) in pairs:
            idx0 = train_bins.get(float(t0))
            idx1 = train_bins.get(float(t1))
            if idx0 is None or idx1 is None or len(idx0) == 0 or len(idx1) == 0:
                continue

            for _ in range(cfg.steps_per_pair_per_epoch):
                B = cfg.batch_size
                sel0 = np.random.choice(idx0, size=B, replace=len(idx0) < B)
                sel1 = np.random.choice(idx1, size=B, replace=len(idx1) < B)

                x0 = torch.from_numpy(X_latent[sel0]).to(device)
                x1 = torch.from_numpy(X_latent[sel1]).to(device)

                t0_t = torch.tensor(float(t0), dtype=torch.float32, device=device)
                t1_t = torch.tensor(float(t1), dtype=torch.float32, device=device)

                y_hat = integrate_ode(
                    lambda xx, tt: model(xx, tt),
                    x0,
                    t0_t,
                    t1_t,
                    cfg.n_steps_per_interval,
                    method=cfg.integrator,
                )

                ot = sinkhorn_ot_loss(y_hat, x1, cfg.ot_epsilon, cfg.sinkhorn_iters)

                t_mid = 0.5 * (t0_t + t1_t)
                f_mid = model(x0, t_mid)
                reg = (f_mid.pow(2).sum(dim=1)).mean()

                loss = ot + cfg.loss_reg_drift * reg

                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                train_ot_losses.append(float(ot.item()))
                train_reg_losses.append(float(reg.item()))

        if not train_ot_losses:
            raise RuntimeError("No training batches were formed – check time bins / splits.")

        train_ot = float(np.mean(train_ot_losses))
        train_reg = float(np.mean(train_reg_losses))
        train_total = train_ot + cfg.loss_reg_drift * train_reg

        # Validation
        val_metrics = evaluate_epoch(model, X_latent, times_sorted, val_bins, cfg)
        val_total = val_metrics["val_total"]
        val_ot = val_metrics["val_ot"]
        val_reg = val_metrics["val_reg"]

        sec = time.time() - t_start

        # Log to CSV
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_total:.6f},{train_ot:.6f},{train_reg:.6f},"
                f"{val_total:.6f},{val_ot:.6f},{val_reg:.6f},{sec:.2f}\n"
            )

        log(
            f"[EPOCH {epoch:03d}] "
            f"train_total={train_total:.4f} (ot={train_ot:.4f}, reg={train_reg:.4f}) "
            f"| val_total={val_total:.4f} (ot={val_ot:.4f}, reg={val_reg:.4f}) "
            f"| {sec:.2f} sec"
        )

        # Early stopping on val_total
        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            no_improve = 0
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                },
                best_path,
            )
            log(f"[CKPT] new best at epoch {epoch}, val_total={best_val:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"[EARLY STOP] no improvement for {patience} epochs, stopping at epoch {epoch}.")
                break

    log(f"[TRAIN] best_val={best_val:.4f} at epoch={best_epoch}")

    # Reload best checkpoint for export
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        log(f"[CKPT] reloaded best checkpoint from {best_path}")
    else:
        log("[WARN] best checkpoint not found; using final model parameters.")

    model.eval()

    # Export vector field samples using best model
    log("[VF] exporting vector field samples from best model ...")
    rng = np.random.RandomState(cfg.seed + 42)
    xs_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for tt in times_sorted:
        tt_f = float(tt)
        idx = bins_all[tt_f]
        if len(idx) == 0:
            continue
        k = min(cfg.vf_samples_per_time, len(idx))
        sel = rng.choice(idx, size=k, replace=False if len(idx) >= k else True)
        xs_list.append(X_latent[sel])
        ts_list.append(np.full((k,), tt_f, dtype=np.float32))

    if xs_list:
        Xs = np.vstack(xs_list).astype(np.float32, copy=False)
        Ts = np.concatenate(ts_list).astype(np.float32, copy=False)
        with torch.no_grad():
            x_t = torch.from_numpy(Xs).to(device)
            t_t = torch.from_numpy(Ts).to(device)
            v = model(x_t, t_t).cpu().numpy().astype(np.float32)
        vf_path = out_dir / "vectorfield_samples.best.npz"
        np.savez_compressed(
            vf_path,
            X_latent=Xs,
            t=Ts,
            v=v,
            times=np.asarray(times_sorted, dtype=np.float32),
        )
        log(f"[DONE] saved vector field samples -> {vf_path}")
    else:
        log("[WARN] no vector field samples exported (no data in bins_all).")

    # Clean
    del adata
    gc.collect()

    log("=== DONE: Step4 dynamics training completed ===")
    log(f"- Logs        : {log_path}")
    log(f"- Best ckpt   : {best_path}")
    log(f"- All ckpts   : {ckpt_dir}")
    log(f"- VF samples  : {out_dir / 'vectorfield_samples.best.npz'}")


# =========================
# CLI
# =========================

def _apply_env_overrides(cfg: Step4Config) -> Step4Config:
    """Keep backward-compatible env overrides (STEP4_*)."""
    if os.environ.get("STEP4_IN_H5AD"):
        cfg.in_h5ad = os.environ["STEP4_IN_H5AD"]
    if os.environ.get("STEP4_OUT_DIR"):
        cfg.out_dir = os.environ["STEP4_OUT_DIR"]
    if os.environ.get("STEP4_EPOCHS"):
        cfg.epochs = int(os.environ["STEP4_EPOCHS"])
    if os.environ.get("STEP4_BATCH"):
        cfg.batch_size = int(os.environ["STEP4_BATCH"])
    if os.environ.get("STEP4_LR"):
        cfg.lr = float(os.environ["STEP4_LR"])
    if os.environ.get("STEP4_DEVICE"):
        cfg.device = os.environ["STEP4_DEVICE"]
    return cfg


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step4 dynamics training (Neural ODE + OT)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON/YAML config file for Step4 (optional).",
    )
    args = parser.parse_args(argv)

    cfg = load_step4_config(args.config)
    cfg = _apply_env_overrides(cfg)

    train_step4(cfg)


if __name__ == "__main__":
    main()
