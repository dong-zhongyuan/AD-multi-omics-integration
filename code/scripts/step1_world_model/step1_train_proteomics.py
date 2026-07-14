#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 训练 World Model (改编自 HepaWorld step4_train.py)
==========================================================

改编要点：
- 保留原版的所有核心逻辑（OT loss, drift reg, early stopping, vector field export）
- 适配血浆→CSF 跨组织映射（不是时间序列，而是空间映射）
- 输入：配对的血浆和 CSF 蛋白组数据
- 输出：训练好的 World Model + vector field samples
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入配置管理器
from tools.config_loader import get_config
config = get_config()

# 添加 HepaWorld 到路径
sys.path.insert(0, str(config.get_path("paths.hepaworld_dir")))

from models.dynamics import DriftNet, integrate_ode
from utils.seed import set_global_seed

# ============================================================================
# 配置
# ============================================================================

@dataclass
class Step1Config:
    """Step 1 配置（改编自 Step4Config）"""
    
    # 路径
    root: str = str(config.get_path("paths.project_root"))
    plasma_h5ad: str = str(config.get_path("paths.processed_data_dir")) + "/plasma_proteomics_paired.h5ad"
    csf_h5ad: str = str(config.get_path("paths.processed_data_dir")) + "/csf_proteomics_paired.h5ad"
    out_dir: str = str(config.get_path("paths.output_dir")) + "/step1_world_model_proteomics"
    
    # 数据预处理（蛋白组：1339 配对样本，132 蛋白）
    protein_dim: int = None  # 将在加载数据后自动设置
    max_proteins: int = None  # 不限制（蛋白数量本身就不多）
    
    # 训练
    seed: int = 42
    deterministic: bool = True
    device: str = "cpu"
    epochs: int = 100
    steps_per_pair_per_epoch: int = 200  # 每个 epoch 的 minibatch 更新次数
    batch_size: int = 128
    
    # ODE 积分器
    integrator: str = "rk4"
    n_steps_per_interval: int = 4
    
    # OT / Loss
    sinkhorn_iters: int = 50
    ot_epsilon: float = 0.15
    loss_reg_drift: float = 1e-4
    
    # 优化器（蛋白组样本充足，使用标准配置）
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # 验证集划分（1339 样本，20% = 267 验证样本）
    val_frac: float = 0.2
    
    # Vector field 采样
    vf_samples_per_tissue: int = 800


# ============================================================================
# Sinkhorn OT Loss（原版）
# ============================================================================

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


# ============================================================================
# 验证函数（改编自原版 evaluate_epoch）
# ============================================================================

def evaluate_epoch(
    model: DriftNet,
    X_plasma_val: np.ndarray,
    X_csf_val: np.ndarray,
    cfg: Step1Config,
    device: torch.device,
) -> Dict[str, float]:
    """计算验证集上的 OT loss + drift regularization"""
    model.eval()
    
    with torch.no_grad():
        x_plasma = torch.from_numpy(X_plasma_val).to(device)
        x_csf = torch.from_numpy(X_csf_val).to(device)
        
        # 血浆 (t=0) → CSF (t=1)
        t0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        t1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # ODE 积分
        y_hat = integrate_ode(
            lambda xx, tt: model(xx, tt),
            x_plasma,
            t0,
            t1,
            cfg.n_steps_per_interval,
            method=cfg.integrator,
        )
        
        # OT loss
        ot = sinkhorn_ot_loss(y_hat, x_csf, cfg.ot_epsilon, cfg.sinkhorn_iters)
        
        # Drift regularization
        t_mid = 0.5 * (t0 + t1)
        f_mid = model(x_plasma, t_mid)
        reg = (f_mid.pow(2).sum(dim=1)).mean()
        
        val_ot = float(ot.item())
        val_reg = float(reg.item())
        val_total = val_ot + cfg.loss_reg_drift * val_reg
    
    return {"val_total": val_total, "val_ot": val_ot, "val_reg": val_reg}


# ============================================================================
# 主训练函数（改编自原版 train_step4）
# ============================================================================

def train_step1(cfg: Step1Config) -> None:
    """主训练入口"""
    
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存配置
    with (out_dir / "config_step1.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    
    # 设置随机种子
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Step 1: 训练 World Model (血浆 → CSF)")
    print("="*60)
    print(f"[INIT] device={device}")
    print(f"[INIT] out_dir={out_dir}")
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n[DATA] 加载配对数据 ...")
    
    import anndata as ad
    
    adata_plasma = ad.read_h5ad(cfg.plasma_h5ad)
    adata_csf = ad.read_h5ad(cfg.csf_h5ad)
    
    print(f"[DATA] 血浆: {adata_plasma.n_obs} 样本 × {adata_plasma.n_vars} 蛋白")
    print(f"[DATA] CSF: {adata_csf.n_obs} 样本 × {adata_csf.n_vars} 蛋白")
    
    # 找共同蛋白
    common_proteins = list(set(adata_plasma.var_names) & set(adata_csf.var_names))
    print(f"[DATA] 共同蛋白数: {len(common_proteins)}")
    
    # 动态设置protein_dim
    if cfg.protein_dim is None:
        cfg.protein_dim = len(common_proteins)
        print(f"[DATA] 自动设置 protein_dim = {cfg.protein_dim}")
    
    # 筛选共同蛋白
    adata_plasma = adata_plasma[:, common_proteins].copy()
    adata_csf = adata_csf[:, common_proteins].copy()
    
    # 提取数据矩阵
    X_plasma = np.asarray(adata_plasma.X, dtype=np.float32)
    X_csf = np.asarray(adata_csf.X, dtype=np.float32)
    
    print(f"[DATA] 最终维度: {X_plasma.shape}")
    
    # ========================================================================
    # 2. 数据标准化
    # ========================================================================
    print("\n[PREP] 数据标准化 ...")
    
    # Z-score 标准化
    X_plasma_mean = X_plasma.mean(axis=0)
    X_plasma_std = X_plasma.std(axis=0) + 1e-8
    X_plasma_norm = (X_plasma - X_plasma_mean) / X_plasma_std
    
    X_csf_mean = X_csf.mean(axis=0)
    X_csf_std = X_csf.std(axis=0) + 1e-8
    X_csf_norm = (X_csf - X_csf_mean) / X_csf_std
    
    print(f"[PREP] 血浆标准化后范围: [{X_plasma_norm.min():.2f}, {X_plasma_norm.max():.2f}]")
    print(f"[PREP] CSF标准化后范围: [{X_csf_norm.min():.2f}, {X_csf_norm.max():.2f}]")
    
    # 保存标准化参数
    np.savez_compressed(
        out_dir / "normalization_params.npz",
        plasma_mean=X_plasma_mean,
        plasma_std=X_plasma_std,
        csf_mean=X_csf_mean,
        csf_std=X_csf_std,
        common_proteins=np.array(common_proteins, dtype=object),
    )
    
    # ========================================================================
    # 3. 划分训练集和验证集
    # ========================================================================
    print("\n[SPLIT] 划分训练集和验证集 ...")
    
    n_samples = X_plasma_norm.shape[0]
    n_train = int(n_samples * (1 - cfg.val_frac))
    
    # 随机打乱
    rng = np.random.RandomState(cfg.seed + 1)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_plasma_train = X_plasma_norm[train_idx]
    X_csf_train = X_csf_norm[train_idx]
    X_plasma_val = X_plasma_norm[val_idx]
    X_csf_val = X_csf_norm[val_idx]
    
    print(f"[SPLIT] 训练集: {len(train_idx)} 样本")
    print(f"[SPLIT] 验证集: {len(val_idx)} 样本")
    
    # ========================================================================
    # 4. 构建模型和优化器
    # ========================================================================
    print("\n[MODEL] 构建模型 ...")
    
    model = DriftNet(
        dim=cfg.protein_dim,
        hidden=128,
        depth=3,
        time_freq=4,
        dropout=0.1
    ).to(device)
    
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] 模型参数数: {n_params:,}")
    
    # ========================================================================
    # 5. 训练循环
    # ========================================================================
    print("\n[TRAIN] 开始训练 ...")
    
    best_val = float("inf")
    best_epoch = -1
    patience = 10
    no_improve = 0
    
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_total,train_ot,train_reg,val_total,val_ot,val_reg,sec\n")
    
    for epoch in range(1, cfg.epochs + 1):
        t_start = time.time()
        model.train()
        
        train_ot_losses: List[float] = []
        train_reg_losses: List[float] = []
        
        # 训练：每个 epoch 做 steps_per_pair_per_epoch 次 minibatch 更新
        for _ in range(cfg.steps_per_pair_per_epoch):
            # 随机采样 batch
            B = cfg.batch_size
            sel = rng.choice(len(train_idx), size=B, replace=len(train_idx) < B)
            
            x0 = torch.from_numpy(X_plasma_train[sel]).to(device)
            x1 = torch.from_numpy(X_csf_train[sel]).to(device)
            
            t0 = torch.tensor(0.0, dtype=torch.float32, device=device)
            t1 = torch.tensor(1.0, dtype=torch.float32, device=device)
            
            # ODE 积分：血浆 → CSF
            y_hat = integrate_ode(
                lambda xx, tt: model(xx, tt),
                x0,
                t0,
                t1,
                cfg.n_steps_per_interval,
                method=cfg.integrator,
            )
            
            # OT loss
            ot = sinkhorn_ot_loss(y_hat, x1, cfg.ot_epsilon, cfg.sinkhorn_iters)
            
            # Drift regularization
            t_mid = 0.5 * (t0 + t1)
            f_mid = model(x0, t_mid)
            reg = (f_mid.pow(2).sum(dim=1)).mean()
            
            # 总损失
            loss = ot + cfg.loss_reg_drift * reg
            
            # 反向传播
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            
            train_ot_losses.append(float(ot.item()))
            train_reg_losses.append(float(reg.item()))
        
        # 计算训练集平均损失
        train_ot = float(np.mean(train_ot_losses))
        train_reg = float(np.mean(train_reg_losses))
        train_total = train_ot + cfg.loss_reg_drift * train_reg
        
        # 验证
        val_metrics = evaluate_epoch(model, X_plasma_val, X_csf_val, cfg, device)
        val_total = val_metrics["val_total"]
        val_ot = val_metrics["val_ot"]
        val_reg = val_metrics["val_reg"]
        
        sec = time.time() - t_start
        
        # 记录到 CSV
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_total:.6f},{train_ot:.6f},{train_reg:.6f},"
                f"{val_total:.6f},{val_ot:.6f},{val_reg:.6f},{sec:.2f}\n"
            )
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[EPOCH {epoch:03d}] "
                f"train_total={train_total:.4f} (ot={train_ot:.4f}, reg={train_reg:.4f}) "
                f"| val_total={val_total:.4f} (ot={val_ot:.4f}, reg={val_reg:.4f}) "
                f"| {sec:.2f} sec"
            )
        
        # Early stopping
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
            if epoch % 10 == 0:
                print(f"[CKPT] 新的最佳模型 at epoch {epoch}, val_total={best_val:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] {patience} 个 epoch 无改进，停止训练")
                break
    
    print(f"\n[TRAIN] 训练完成！最佳 val_total={best_val:.4f} at epoch={best_epoch}")
    
    # ========================================================================
    # 6. 重新加载最佳模型并导出 vector field
    # ========================================================================
    print("\n[VF] 导出 vector field samples ...")
    
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[CKPT] 已加载最佳模型: {best_path}")
    else:
        print("[WARN] 未找到最佳模型，使用最终模型参数")
    
    model.eval()
    
    # 采样 vector field：从血浆和 CSF 各采样一些点
    rng_vf = np.random.RandomState(cfg.seed + 42)
    
    # 血浆 (t=0)
    k_plasma = min(cfg.vf_samples_per_tissue, len(X_plasma_norm))
    sel_plasma = rng_vf.choice(len(X_plasma_norm), size=k_plasma, replace=False)
    Xs_plasma = X_plasma_norm[sel_plasma]
    Ts_plasma = np.zeros(k_plasma, dtype=np.float32)
    
    # CSF (t=1)
    k_csf = min(cfg.vf_samples_per_tissue, len(X_csf_norm))
    sel_csf = rng_vf.choice(len(X_csf_norm), size=k_csf, replace=False)
    Xs_csf = X_csf_norm[sel_csf]
    Ts_csf = np.ones(k_csf, dtype=np.float32)
    
    # 合并
    Xs = np.vstack([Xs_plasma, Xs_csf]).astype(np.float32)
    Ts = np.concatenate([Ts_plasma, Ts_csf]).astype(np.float32)
    
    # 计算 vector field
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
        tissues=np.array(["plasma", "csf"], dtype=object),
    )
    print(f"[VF] 已保存 vector field samples: {vf_path}")
    
    # ========================================================================
    # 7. 清理
    # ========================================================================
    del adata_plasma, adata_csf
    gc.collect()
    
    print("\n" + "="*60)
    print("✅ Step 1 完成！")
    print("="*60)
    print(f"- 训练日志: {log_path}")
    print(f"- 最佳模型: {best_path}")
    print(f"- Vector field: {vf_path}")
    print(f"- 输出目录: {out_dir}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    cfg = Step1Config()
    train_step1(cfg)


if __name__ == "__main__":
    main()
