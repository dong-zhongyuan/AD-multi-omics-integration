#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 训练 World Model - 转录组版本
==========================================================

改编要点：
- 保留原版的所有核心逻辑（OT loss, drift reg, early stopping, vector field export）
- 适配血液→脑 跨组织映射（不是时间序列，而是空间映射）
- 输入：配对的血液和脑转录组数据
- 输出：训练好的 World Model + vector field samples
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import scipy.sparse as sp

# CPU 多线程加速（16核全开）
torch.set_num_threads(min(16, os.cpu_count() or 4))
torch.set_num_interop_threads(4)

# 添加项目根目录到路径（自动适配 WSL / Windows）
_win_root = Path(__file__).resolve().parents[2]
_wsl_root = Path(str(PROJECT_ROOT))
PROJECT_ROOT = _wsl_root if _wsl_root.exists() and _wsl_root.is_dir() and str(_wsl_root.resolve()).startswith('/mnt/') else _win_root
sys.path.insert(0, str(PROJECT_ROOT))

# 导入配置管理器
from tools.config_loader import get_config
config = get_config()

# 添加 HepaWorld 到路径
_HEPAWORLD = Path(str(config.get_path("paths.hepaworld_dir")))
if not _HEPAWORLD.exists():
    _HEPAWORLD = PROJECT_ROOT / "tools" / "hepaworld"
sys.path.insert(0, str(_HEPAWORLD))

from models.dynamics import DriftNet, integrate_ode
from utils.seed import set_global_seed


def _dense_backed_slice(X, row_idx):
    """Convert a backed h5ad row slice to a dense float32 numpy array."""
    block = X[row_idx, :]
    if sp.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)

# ============================================================================
# 配置
# ============================================================================

def _fix_path(p):
    """config 返回的 /mnt/d/ 路径在 Windows 下不可用，转成 PROJECT_ROOT 相对路径。"""
    s = str(p).replace("\\", "/")
    if "." in s:
        rel = s.split(".", 1)[1]
        return str(PROJECT_ROOT / rel.lstrip("/"))
    return s


@dataclass
class Step1Config:
    """Step 1 配置（改编自 Step4Config）"""

    # 路径（_fix_path 确保 Windows 下也能用）
    root: str = str(PROJECT_ROOT)
    plasma_h5ad: str = str(PROJECT_ROOT / "processed-data" / "transcriptomics_blood.h5ad")
    csf_h5ad: str = str(PROJECT_ROOT / "processed-data" / "transcriptomics_brain.h5ad")
    out_dir: str = str(PROJECT_ROOT / "output" / "step1_world_model_transcriptomics_no_pca")
    
    # 数据预处理（直接用原始基因表达，不做 PCA 降维）
    # gene_dim 将在运行时动态设置为共同基因数
    gene_dim: int = None  # 将在加载数据后自动设置
    
    # 内存优化参数
    use_sparse: bool = False  # 是否使用稀疏矩阵（如果数据稀疏）
    max_genes: int = 2000  # 最大基因数限制（适配大样本量：19921血液+21292脑）
    
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
    
    # 优化器（小样本场景：增强正则化）
    lr: float = 2e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    
    # 验证集划分（大样本场景：血液19921样本，脑21292样本）
    val_frac: float = 0.1  # 验证集比例（10%足够评估）
    
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
    adata_plasma,
    adata_csf,
    plasma_val_idx: np.ndarray,
    csf_val_idx: np.ndarray,
    plasma_gene_indices: list,
    csf_gene_indices: list,
    X_plasma_mean: np.ndarray,
    X_plasma_std: np.ndarray,
    X_csf_mean: np.ndarray,
    X_csf_std: np.ndarray,
    cfg: Step1Config,
    device: torch.device,
) -> Dict[str, float]:
    """计算验证集上的 OT loss + drift regularization（流式处理）"""
    model.eval()

    with torch.no_grad():
        # 验证集已经是 batch_size 大小，直接使用
        plasma_sorted_order = np.argsort(plasma_val_idx)
        plasma_val_idx_sorted = plasma_val_idx[plasma_sorted_order]
        
        csf_sorted_order = np.argsort(csf_val_idx)
        csf_val_idx_sorted = csf_val_idx[csf_sorted_order]
        
        # 读取数据（递增索引）
        x_plasma_full = _dense_backed_slice(adata_plasma.X, plasma_val_idx_sorted)
        x_plasma_raw = x_plasma_full[:, plasma_gene_indices]

        x_csf_full = _dense_backed_slice(adata_csf.X, csf_val_idx_sorted)
        x_csf_raw = x_csf_full[:, csf_gene_indices]
        
        # 恢复原始顺序
        x_plasma_raw = x_plasma_raw[np.argsort(plasma_sorted_order)]
        x_csf_raw = x_csf_raw[np.argsort(csf_sorted_order)]
        
        # Z-score 标准化
        x_plasma_norm = (x_plasma_raw - X_plasma_mean) / X_plasma_std
        x_csf_norm = (x_csf_raw - X_csf_mean) / X_csf_std
        
        x_plasma = torch.from_numpy(x_plasma_norm).to(device)
        x_csf = torch.from_numpy(x_csf_norm).to(device)
        
        # 清理临时数组
        del x_plasma_raw, x_csf_raw, x_plasma_norm, x_csf_norm, x_plasma_full, x_csf_full
        
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
    print("Step 1: 训练 World Model (血液 → 脑)")
    print("="*60)
    print(f"[INIT] device={device}")
    print(f"[INIT] out_dir={out_dir}")
    
    # ========================================================================
    # 1. 加载数据（backed模式，不加载到内存）
    # ========================================================================
    print("\n[DATA] 加载配对数据（backed模式）...")
    
    import anndata as ad
    
    # backed='r' 模式：只加载元数据，不加载表达矩阵到内存
    adata_plasma = ad.read_h5ad(cfg.plasma_h5ad, backed='r')
    adata_csf = ad.read_h5ad(cfg.csf_h5ad, backed='r')
    
    print(f"[DATA] 血液: {adata_plasma.n_obs} 样本 × {adata_plasma.n_vars} 基因")
    print(f"[DATA] 脑: {adata_csf.n_obs} 样本 × {adata_csf.n_vars} 基因")
    
    # 找共同基因
    common_proteins = list(set(adata_plasma.var_names) & set(adata_csf.var_names))
    print(f"[DATA] 共同基因数: {len(common_proteins)}")
    
    # 获取共同基因在原始数据中的索引
    plasma_gene_indices = [list(adata_plasma.var_names).index(g) for g in common_proteins]
    csf_gene_indices = [list(adata_csf.var_names).index(g) for g in common_proteins]
    
    # 如果设置了最大基因数限制，进行 HVG 特征选择（分块计算方差）
    if cfg.max_genes is not None and len(common_proteins) > cfg.max_genes:
        print(f"[DATA] 基因数 {len(common_proteins)} 超过限制 {cfg.max_genes}，选择高变异基因 (HVG)...")
        print(f"[DATA] 使用分块计算方差，避免内存溢出...")
        
        chunk_size = 1000  # 每次读取1000个样本
        n_genes = len(common_proteins)
        
        # 初始化累加器
        sum_plasma = np.zeros(n_genes, dtype=np.float32)
        sum_sq_plasma = np.zeros(n_genes, dtype=np.float32)
        count_plasma = 0
        
        sum_csf = np.zeros(n_genes, dtype=np.float32)
        sum_sq_csf = np.zeros(n_genes, dtype=np.float32)
        count_csf = 0
        
        # 分块读取血液数据计算方差
        print(f"[DATA] 计算血液数据方差...")
        for start_idx in range(0, adata_plasma.n_obs, chunk_size):
            end_idx = min(start_idx + chunk_size, adata_plasma.n_obs)
            # h5py要求索引有序，先读整行再用numpy索引选列
            chunk_full = _dense_backed_slice(adata_plasma.X, slice(start_idx, end_idx))
            chunk = chunk_full[:, plasma_gene_indices]
            sum_plasma += chunk.sum(axis=0)
            sum_sq_plasma += (chunk ** 2).sum(axis=0)
            count_plasma += chunk.shape[0]
            del chunk, chunk_full
            gc.collect()
        
        # 分块读取脑数据计算方差
        print(f"[DATA] 计算脑数据方差...")
        for start_idx in range(0, adata_csf.n_obs, chunk_size):
            end_idx = min(start_idx + chunk_size, adata_csf.n_obs)
            # h5py要求索引有序，先读整行再用numpy索引选列
            chunk_full = _dense_backed_slice(adata_csf.X, slice(start_idx, end_idx))
            chunk = chunk_full[:, csf_gene_indices]
            sum_csf += chunk.sum(axis=0)
            sum_sq_csf += (chunk ** 2).sum(axis=0)
            count_csf += chunk.shape[0]
            del chunk, chunk_full
            gc.collect()
        
        # 计算方差: Var(X) = E[X^2] - E[X]^2
        mean_plasma = sum_plasma / count_plasma
        var_plasma = (sum_sq_plasma / count_plasma) - (mean_plasma ** 2)
        
        mean_csf = sum_csf / count_csf
        var_csf = (sum_sq_csf / count_csf) - (mean_csf ** 2)
        
        # 选择 top max_genes 个高方差基因
        var_combined = var_plasma + var_csf
        top_indices = np.argsort(var_combined)[-cfg.max_genes:]
        
        # 筛选高变异基因
        hvg_genes = [common_proteins[i] for i in top_indices]
        plasma_gene_indices = [plasma_gene_indices[i] for i in top_indices]
        csf_gene_indices = [csf_gene_indices[i] for i in top_indices]
        common_proteins = hvg_genes
        
        print(f"[DATA] HVG 筛选后: {len(common_proteins)} 基因")
        
        # 清理
        del sum_plasma, sum_sq_plasma, sum_csf, sum_sq_csf
        del mean_plasma, mean_csf, var_plasma, var_csf, var_combined
        gc.collect()
    
    # 动态设置 gene_dim
    cfg.gene_dim = len(common_proteins)
    print(f"[DATA] 最终 gene_dim = {cfg.gene_dim}")
    print(f"[DATA] 血液样本数: {adata_plasma.n_obs}, 脑样本数: {adata_csf.n_obs}")
    
    # ========================================================================
    # 2. 数据标准化（分块计算均值和标准差）
    # ========================================================================
    print("\n[PREP] 数据标准化（分块计算）...")
    
    chunk_size = 1000
    n_genes = len(common_proteins)
    
    # 计算血液数据的均值和标准差
    print(f"[PREP] 计算血液数据统计量...")
    sum_plasma = np.zeros(n_genes, dtype=np.float32)
    sum_sq_plasma = np.zeros(n_genes, dtype=np.float32)
    count_plasma = 0
    
    for start_idx in range(0, adata_plasma.n_obs, chunk_size):
        end_idx = min(start_idx + chunk_size, adata_plasma.n_obs)
        # h5py要求索引有序，先读整行再用numpy索引选列
        chunk_full = _dense_backed_slice(adata_plasma.X, slice(start_idx, end_idx))
        chunk = chunk_full[:, plasma_gene_indices]
        sum_plasma += chunk.sum(axis=0)
        sum_sq_plasma += (chunk ** 2).sum(axis=0)
        count_plasma += chunk.shape[0]
        del chunk, chunk_full
        gc.collect()
    
    X_plasma_mean = sum_plasma / count_plasma
    X_plasma_std = np.sqrt((sum_sq_plasma / count_plasma) - (X_plasma_mean ** 2)) + 1e-8
    
    del sum_plasma, sum_sq_plasma
    gc.collect()
    
    # 计算脑数据的均值和标准差
    print(f"[PREP] 计算脑数据统计量...")
    sum_csf = np.zeros(n_genes, dtype=np.float32)
    sum_sq_csf = np.zeros(n_genes, dtype=np.float32)
    count_csf = 0
    
    for start_idx in range(0, adata_csf.n_obs, chunk_size):
        end_idx = min(start_idx + chunk_size, adata_csf.n_obs)
        # h5py要求索引有序，先读整行再用numpy索引选列
        chunk_full = _dense_backed_slice(adata_csf.X, slice(start_idx, end_idx))
        chunk = chunk_full[:, csf_gene_indices]
        sum_csf += chunk.sum(axis=0)
        sum_sq_csf += (chunk ** 2).sum(axis=0)
        count_csf += chunk.shape[0]
        del chunk, chunk_full
        gc.collect()
    
    X_csf_mean = sum_csf / count_csf
    X_csf_std = np.sqrt((sum_sq_csf / count_csf) - (X_csf_mean ** 2)) + 1e-8
    
    del sum_csf, sum_sq_csf
    gc.collect()
    
    print(f"[PREP] 血液: mean范围=[{X_plasma_mean.min():.4f}, {X_plasma_mean.max():.4f}], std范围=[{X_plasma_std.min():.4f}, {X_plasma_std.max():.4f}]")
    print(f"[PREP] 脑: mean范围=[{X_csf_mean.min():.4f}, {X_csf_mean.max():.4f}], std范围=[{X_csf_std.min():.4f}, {X_csf_std.max():.4f}]")
    
    # 保存标准化参数
    np.savez_compressed(
        out_dir / "normalization_params.npz",
        plasma_mean=X_plasma_mean,
        plasma_std=X_plasma_std,
        csf_mean=X_csf_mean,
        csf_std=X_csf_std,
        common_genes=np.array(common_proteins, dtype=object),
    )
    
    # ========================================================================
    # 3. 划分训练集和验证集
    # ========================================================================
    n_val = cfg.batch_size  # 验证集大小 = batch_size（与训练 loss 可比）
    print(f"\n[SPLIT] Blood (Bone Marrow): {adata_plasma.n_obs} cells → 训练 {adata_plasma.n_obs - n_val} / 验证 {n_val}")
    print(f"[SPLIT] Brain: {adata_csf.n_obs} cells → 训练 {adata_csf.n_obs - n_val} / 验证 {n_val}")

    rng = np.random.RandomState(cfg.seed + 1)

    # Blood 划分
    plasma_indices = rng.permutation(adata_plasma.n_obs)
    plasma_val_idx = np.sort(plasma_indices[:n_val])
    plasma_train_idx = np.sort(plasma_indices[n_val:])

    # Brain 划分
    csf_indices = rng.permutation(adata_csf.n_obs)
    csf_val_idx = np.sort(csf_indices[:n_val])
    csf_train_idx = np.sort(csf_indices[n_val:])
    
    # ========================================================================
    # 4. 构建模型和优化器
    # ========================================================================
    print("\n[MODEL] 构建模型 ...")
    
    # 小样本场景：降低模型复杂度，增强正则化
    hidden_dim = 64
    depth = 2
    dropout = 0.3
    
    print(f"[MODEL] 模型配置: hidden={hidden_dim}, depth={depth}, dropout={dropout}")
    
    model = DriftNet(
        dim=cfg.gene_dim,
        hidden=hidden_dim,
        depth=depth,
        time_freq=4,
        dropout=dropout
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
        for step_idx in range(cfg.steps_per_pair_per_epoch):
            # 随机采样 batch（血液和脑独立采样）
            B = cfg.batch_size
            sel_plasma = rng.choice(len(plasma_train_idx), size=B, replace=len(plasma_train_idx) < B)
            sel_csf = rng.choice(len(csf_train_idx), size=B, replace=len(csf_train_idx) < B)
            
            # 从backed模式读取数据并标准化
            plasma_batch_idx = plasma_train_idx[sel_plasma]
            csf_batch_idx = csf_train_idx[sel_csf]
            
            # h5py要求行索引也必须递增，先排序再读取
            plasma_sorted_order = np.argsort(plasma_batch_idx)
            plasma_batch_idx_sorted = plasma_batch_idx[plasma_sorted_order]
            
            csf_sorted_order = np.argsort(csf_batch_idx)
            csf_batch_idx_sorted = csf_batch_idx[csf_sorted_order]
            
            # 读取数据（递增索引）
            x0_full = _dense_backed_slice(adata_plasma.X, plasma_batch_idx_sorted)
            x0_raw = x0_full[:, plasma_gene_indices]

            x1_full = _dense_backed_slice(adata_csf.X, csf_batch_idx_sorted)
            x1_raw = x1_full[:, csf_gene_indices]
            
            # 恢复原始顺序
            x0_raw = x0_raw[np.argsort(plasma_sorted_order)]
            x1_raw = x1_raw[np.argsort(csf_sorted_order)]
            
            # Z-score 标准化
            x0_norm = (x0_raw - X_plasma_mean) / X_plasma_std
            x1_norm = (x1_raw - X_csf_mean) / X_csf_std
            
            x0 = torch.from_numpy(x0_norm).to(device)
            x1 = torch.from_numpy(x1_norm).to(device)
            
            # 清理临时数组
            del x0_raw, x1_raw, x0_norm, x1_norm, x0_full, x1_full
            
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
            
            # 显式释放中间变量
            del x0, x1, y_hat, ot, reg, loss, f_mid
            
            # 每 50 步清理一次内存
            if step_idx % 50 == 0:
                gc.collect()
        
        # 计算训练集平均损失
        train_ot = float(np.mean(train_ot_losses))
        train_reg = float(np.mean(train_reg_losses))
        train_total = train_ot + cfg.loss_reg_drift * train_reg
        
        # 验证（每个epoch都验证）
        val_metrics = evaluate_epoch(
            model,
            adata_plasma,
            adata_csf,
            plasma_val_idx,
            csf_val_idx,
            plasma_gene_indices,
            csf_gene_indices,
            X_plasma_mean,
            X_plasma_std,
            X_csf_mean,
            X_csf_std,
            cfg,
            device
        )
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
        
        # 打印进度（每个 epoch 都输出）
        if True:
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
    
    # 采样 vector field：从血浆和 CSF 各采样一些点（流式处理）
    rng_vf = np.random.RandomState(cfg.seed + 42)
    
    # 血液 (t=0)
    k_plasma = min(cfg.vf_samples_per_tissue, adata_plasma.n_obs)
    sel_plasma_idx = rng_vf.choice(adata_plasma.n_obs, size=k_plasma, replace=False)
    
    # 从backed模式读取并标准化
    # h5py要求行索引也必须递增，先排序再读取
    plasma_sorted_order = np.argsort(sel_plasma_idx)
    sel_plasma_idx_sorted = sel_plasma_idx[plasma_sorted_order]
    
    Xs_plasma_full = _dense_backed_slice(adata_plasma.X, sel_plasma_idx_sorted)
    Xs_plasma_raw = Xs_plasma_full[:, plasma_gene_indices]
    
    # 恢复原始顺序
    Xs_plasma_raw = Xs_plasma_raw[np.argsort(plasma_sorted_order)]
    Xs_plasma = (Xs_plasma_raw - X_plasma_mean) / X_plasma_std
    del Xs_plasma_raw, Xs_plasma_full
    
    Ts_plasma = np.zeros(k_plasma, dtype=np.float32)
    
    # 脑 (t=1)
    k_csf = min(cfg.vf_samples_per_tissue, adata_csf.n_obs)
    sel_csf_idx = rng_vf.choice(adata_csf.n_obs, size=k_csf, replace=False)
    
    # 从backed模式读取并标准化
    # h5py要求行索引也必须递增，先排序再读取
    csf_sorted_order = np.argsort(sel_csf_idx)
    sel_csf_idx_sorted = sel_csf_idx[csf_sorted_order]
    
    Xs_csf_full = _dense_backed_slice(adata_csf.X, sel_csf_idx_sorted)
    Xs_csf_raw = Xs_csf_full[:, csf_gene_indices]
    
    # 恢复原始顺序
    Xs_csf_raw = Xs_csf_raw[np.argsort(csf_sorted_order)]
    Xs_csf = (Xs_csf_raw - X_csf_mean) / X_csf_std
    del Xs_csf_raw, Xs_csf_full
    
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
        tissues=np.array(["blood", "brain"], dtype=object),
    )
    print(f"[VF] 已保存 vector field samples: {vf_path}")
    
    # ========================================================================
    # 7. 清理（关闭backed文件）
    # ========================================================================
    print("\n[CLEANUP] 关闭backed文件...")
    adata_plasma.file.close()
    adata_csf.file.close()
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
