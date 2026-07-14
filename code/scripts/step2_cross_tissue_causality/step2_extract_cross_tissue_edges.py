#!/usr/bin/env python3
"""
Step 2: 提取跨组织因果边
=====================================

目标：
- 基于 step1 的世界模型，识别跨组织因果关系
- 识别组织间的相互影响关系

方法：
- 计算 Jacobian 矩阵
- 显著的梯度表示因果边
- 使用 permutation test 确定显著性阈值
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 导入配置管理器
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config
config = get_config()

# 添加 HepaWorld 到路径（自动适配 Windows）
_HEPAWORLD = Path(str(config.get_path("paths.hepaworld_dir")))
if not _HEPAWORLD.exists():
    _HEPAWORLD = PROJECT_ROOT / "tools" / "hepaworld"
sys.path.insert(0, str(_HEPAWORLD))
from models.dynamics import DriftNet, integrate_ode

# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OMICS_CONFIG = {
    'metabolomics': {
        'model_dir': PROJECT_ROOT / 'output/step1_world_model_metabolomics',
        'data_plasma': PROJECT_ROOT / 'processed-data/plasma_metabolomics_common.h5ad',
        'data_csf': PROJECT_ROOT / 'processed-data/csf_metabolomics_common.h5ad',
    },
    'proteomics': {
        'model_dir': PROJECT_ROOT / 'output/step1_world_model_proteomics',
        'data_plasma': PROJECT_ROOT / 'processed-data/plasma_proteomics_paired.h5ad',
        'data_csf': PROJECT_ROOT / 'processed-data/csf_proteomics_paired.h5ad',
    },
    'transcriptomics': {
        'model_dir': PROJECT_ROOT / 'output/step1_world_model_transcriptomics_no_pca',
        'data_plasma': PROJECT_ROOT / 'processed-data/transcriptomics_blood.h5ad',
        'data_csf': PROJECT_ROOT / 'processed-data/transcriptomics_brain.h5ad',
    },
}

OUTPUT_DIR = PROJECT_ROOT / "output/step2_cross_tissue_causality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 核心函数
# ============================================================================

def load_model_and_data(omics_type):
    """加载世界模型和数据"""
    import anndata as ad
    
    config = OMICS_CONFIG[omics_type]
    model_dir = Path(config['model_dir'])
    
    print(f"\n[{omics_type.upper()}] 加载模型和数据...")
    
    # 1. 加载配置
    with open(model_dir / 'config_step1.json') as f:
        cfg = json.load(f)
    
    # 2. 加载数据（backed模式，不加载到内存）
    adata_plasma = ad.read_h5ad(config['data_plasma'], backed='r')
    adata_csf = ad.read_h5ad(config['data_csf'], backed='r')
    
    # 3. 加载标准化参数（先加载，确定训练时用的特征）
    norm_params = np.load(model_dir / 'normalization_params.npz', allow_pickle=True)
    plasma_mean = norm_params['plasma_mean']
    plasma_std = norm_params['plasma_std']
    csf_mean = norm_params['csf_mean']
    csf_std = norm_params['csf_std']
    
    # 获取训练时用的特征名（尝试多种可能的键名）
    trained_features = None
    for key in ['common_metabolites', 'common_proteins', 'common_genes', 'selected_genes', 'feature_names']:
        if key in norm_params:
            trained_features = list(norm_params[key])
            print(f"  从 {key} 读取到 {len(trained_features)} 个特征")
            break
    
    if trained_features is None:
        raise ValueError(f"无法从 normalization_params.npz 中找到特征名，可用键: {list(norm_params.keys())}")
    
    # 获取特征索引（plasma / CSF 的列顺序可能不同，必须分别映射）
    plasma_var_names = list(adata_plasma.var_names)
    csf_var_names = list(adata_csf.var_names)
    plasma_feature_indices = np.asarray([plasma_var_names.index(f) for f in trained_features], dtype=np.int64)
    csf_feature_indices = np.asarray([csf_var_names.index(f) for f in trained_features], dtype=np.int64)
    
    print(f"  血浆: {adata_plasma.n_obs} 样本 × {len(trained_features)} 特征")
    print(f"  CSF: {adata_csf.n_obs} 样本 × {len(trained_features)} 特征")
    
    # 4. 准备标准化参数字典
    norm_dict = {
        'plasma_mean': plasma_mean,
        'plasma_std': plasma_std,
        'csf_mean': csf_mean,
        'csf_std': csf_std,
    }
    
    common_features = trained_features
    
    # 5. 加载模型
    checkpoint_path = model_dir / 'checkpoints' / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 推断模型参数
    state_dict = checkpoint['state_dict']
    feature_dim = len(trained_features)
    
    # 从 state_dict 推断 hidden_dim 和 depth
    mlp_keys = [k for k in state_dict.keys() if k.startswith('mlp.')]
    
    # 获取所有层索引
    layer_indices = sorted(set(int(k.split('.')[1]) for k in mlp_keys if k.split('.')[1].isdigit()))
    
    # 检查是否有 dropout（看索引间隔）
    if len(layer_indices) >= 2:
        gap = layer_indices[1] - layer_indices[0]
        has_dropout = (gap == 3)
    else:
        has_dropout = False
    
    # depth = 层数
    depth = len(layer_indices)
    
    # 获取 hidden_dim
    first_linear_weight = state_dict['mlp.0.weight']
    hidden_dim = first_linear_weight.shape[0]
    
    print(f"  模型参数: feature_dim={feature_dim}, hidden_dim={hidden_dim}, depth={depth}, dropout={has_dropout}")
    
    # 6. 创建模型
    if has_dropout:
        model = DriftNet(feature_dim, hidden_dim, depth, dropout=0.1)
    else:
        model = DriftNet(feature_dim, hidden_dim, depth)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 7. 加载并标准化数据（2000个HVG，数据量不大，可以一次性加载）
    print(f"  加载数据到内存...")
    
    # h5py backed 数据要求 fancy indexing 的列索引单调递增，先排序读取，再恢复训练顺序
    plasma_sorted_order = np.argsort(plasma_feature_indices)
    plasma_sorted_indices = plasma_feature_indices[plasma_sorted_order]
    X_plasma_raw = adata_plasma.X[:, plasma_sorted_indices].toarray() if hasattr(adata_plasma.X[:, plasma_sorted_indices], 'toarray') else np.array(adata_plasma.X[:, plasma_sorted_indices])
    X_plasma_raw = X_plasma_raw[:, np.argsort(plasma_sorted_order)]
    
    csf_sorted_order = np.argsort(csf_feature_indices)
    csf_sorted_indices = csf_feature_indices[csf_sorted_order]
    X_csf_raw = adata_csf.X[:, csf_sorted_indices].toarray() if hasattr(adata_csf.X[:, csf_sorted_indices], 'toarray') else np.array(adata_csf.X[:, csf_sorted_indices])
    X_csf_raw = X_csf_raw[:, np.argsort(csf_sorted_order)]
    
    # 标准化
    X_plasma = (X_plasma_raw - plasma_mean) / (plasma_std + 1e-8)
    X_csf = (X_csf_raw - csf_mean) / (csf_std + 1e-8)
    
    # 转为float32节省内存
    X_plasma = X_plasma.astype(np.float32)
    X_csf = X_csf.astype(np.float32)

    # 下采样以加速 Jacobian 计算（Jacobian 在分布均值处计算，2000 细胞足够）
    MAX_JACOBIAN_SAMPLES = 2000
    rng_ds = np.random.RandomState(42)
    if X_plasma.shape[0] > MAX_JACOBIAN_SAMPLES:
        idx = rng_ds.choice(X_plasma.shape[0], MAX_JACOBIAN_SAMPLES, replace=False)
        X_plasma = X_plasma[idx]
        print(f"  下采样 plasma: {MAX_JACOBIAN_SAMPLES} cells (from {adata_plasma.n_obs})")
    if X_csf.shape[0] > MAX_JACOBIAN_SAMPLES:
        idx = rng_ds.choice(X_csf.shape[0], MAX_JACOBIAN_SAMPLES, replace=False)
        X_csf = X_csf[idx]
        print(f"  下采样 CSF: {MAX_JACOBIAN_SAMPLES} cells (from {adata_csf.n_obs})")

    # 关闭backed文件
    adata_plasma.file.close()
    adata_csf.file.close()
    
    print(f"  数据加载完成: plasma {X_plasma.shape}, CSF {X_csf.shape}")
    
    return model, X_plasma, X_csf, common_features, cfg


def compute_jacobian_brain_to_blood(model, X_csf, cfg, batch_size=64):
    """
    计算反向 Jacobian 矩阵（批处理优化版本）
    
    使用批处理和在线算法累积统计量，大幅提升计算速度
    
    Args:
        model: 世界模型
        X_csf: CSF 数据 (n_samples, n_features)
        cfg: 配置
        batch_size: 批处理大小（默认64）
    
    Returns:
        jacobian_mean: (n_features, n_features) 平均 Jacobian 矩阵
        jacobian_std: (n_features, n_features) Jacobian 标准差矩阵
        n_samples: 样本数
    """
    print(f"\n  计算反向 Jacobian（批处理优化，batch_size={batch_size}）...")
    
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    n_samples, n_features = X_csf.shape
    
    # 使用在线算法累积统计量
    jacobian_sum = np.zeros((n_features, n_features), dtype=np.float64)
    jacobian_sum_sq = np.zeros((n_features, n_features), dtype=np.float64)
    
    # 批处理计算
    n_batches = (n_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc="  批次"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        current_batch_size = end - start
        
        # 准备批次数据
        x_batch = torch.tensor(X_csf[start:end], dtype=torch.float32, device=device, requires_grad=True)
        
        # 反向积分 (t=1 → t=0)
        t1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        t0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # ODE 反向积分（批量）
        y_batch = integrate_ode(
            lambda xx, tt: model(xx, tt),
            x_batch,
            t1,
            t0,
            cfg.get('n_steps_per_interval', 4),
            method=cfg.get('integrator', 'rk4'),
        )
        
        for j in range(n_features):
            # 清空梯度
            if x_batch.grad is not None:
                x_batch.grad.zero_()
            
            # 对所有样本的第j个输出求梯度
            grad_outputs = torch.zeros_like(y_batch)
            grad_outputs[:, j] = 1.0
            
            grads = torch.autograd.grad(
                outputs=y_batch,
                inputs=x_batch,
                grad_outputs=grad_outputs,
                retain_graph=(j < n_features - 1),  # 最后一个不需要保留图
                create_graph=False,
            )[0]

            grads_np = grads.cpu().numpy()
            jacobian_sum[j, :] += grads_np.sum(axis=0, dtype=np.float64)
            jacobian_sum_sq[j, :] += np.square(grads_np, dtype=np.float32).sum(axis=0, dtype=np.float64)
    
    # 计算均值和标准差（裁剪浮点误差导致的极小负方差）
    jacobian_mean64 = jacobian_sum / n_samples
    jacobian_var = np.maximum((jacobian_sum_sq / n_samples) - (jacobian_mean64 ** 2), 0.0)
    jacobian_mean = jacobian_mean64.astype(np.float32)
    jacobian_std = np.sqrt(jacobian_var).astype(np.float32)
    
    return jacobian_mean, jacobian_std, n_samples


def compute_intra_tissue_jacobian(model, X_tissue, cfg, tissue_name, batch_size=64):
    """
    计算单组织内部的 Jacobian 矩阵（批处理优化版本）
    
    使用极小时间步长（dt=0.01）计算瞬时梯度，捕捉组织内部的调控关系
    使用批处理和在线算法累积统计量，大幅提升计算速度
    
    Args:
        model: 世界模型
        X_tissue: 组织数据 (n_samples, n_features)
        cfg: 配置
        tissue_name: 组织名称（用于日志）
        batch_size: 批处理大小（默认64）
    
    Returns:
        jacobian_mean: (n_features, n_features) 平均 Jacobian 矩阵
        jacobian_std: (n_features, n_features) Jacobian 标准差矩阵
        n_samples: 样本数
    """
    print(f"\n  计算 {tissue_name} 内部网络 Jacobian（批处理优化，batch_size={batch_size}）...")
    
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    n_samples, n_features = X_tissue.shape
    
    # 初始化累积统计量
    jacobian_sum = np.zeros((n_features, n_features), dtype=np.float64)
    jacobian_sum_sq = np.zeros((n_features, n_features), dtype=np.float64)
    
    print(f"    样本数: {n_samples}, 特征数: {n_features}")
    print(f"    内存需求: 累积统计量 ~{2 * n_features * n_features * 8 / 1024**2:.1f} MB")
    
    # 批处理计算
    n_batches = (n_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc=f"  {tissue_name} 批次"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        current_batch_size = end - start
        
        # 准备批次数据
        x_batch = torch.tensor(X_tissue[start:end], dtype=torch.float32, device=device, requires_grad=True)
        
        # 使用极小时间步长捕捉瞬时动力学
        t_start = torch.tensor(0.0 if tissue_name == 'blood' else 1.0, dtype=torch.float32, device=device)
        t_end = t_start + 0.01  # 极小时间步长
        
        # ODE 积分（批量）
        y_batch = integrate_ode(
            lambda xx, tt: model(xx, tt),
            x_batch,
            t_start,
            t_end,
            cfg.get('n_steps_per_interval', 4),
            method=cfg.get('integrator', 'rk4'),
        )
        
        for j in range(n_features):
            # 清空梯度
            if x_batch.grad is not None:
                x_batch.grad.zero_()
            
            # 对所有样本的第j个输出求梯度
            grad_outputs = torch.zeros_like(y_batch)
            grad_outputs[:, j] = 1.0
            
            grads = torch.autograd.grad(
                outputs=y_batch,
                inputs=x_batch,
                grad_outputs=grad_outputs,
                retain_graph=(j < n_features - 1),
                create_graph=False,
            )[0]

            grads_np = grads.cpu().numpy()
            jacobian_sum[j, :] += grads_np.sum(axis=0, dtype=np.float64)
            jacobian_sum_sq[j, :] += np.square(grads_np, dtype=np.float32).sum(axis=0, dtype=np.float64)
    
    # 计算均值和标准差（裁剪浮点误差导致的极小负方差）
    jacobian_mean64 = jacobian_sum / n_samples
    jacobian_var = np.maximum((jacobian_sum_sq / n_samples) - (jacobian_mean64 ** 2), 0.0)
    jacobian_mean = jacobian_mean64.astype(np.float32)
    jacobian_std = np.sqrt(jacobian_var).astype(np.float32)
    
    print(f"  {tissue_name} Jacobian 计算完成")
    print(f"    均值范围: [{jacobian_mean.min():.6f}, {jacobian_mean.max():.6f}]")
    print(f"    标准差范围: [{jacobian_std.min():.6f}, {jacobian_std.max():.6f}]")
    
    return jacobian_mean, jacobian_std, n_samples


def identify_significant_edges(jacobian_mean, jacobian_std, n_samples, feature_names, threshold_percentile=95, direction='brain_to_blood'):
    """
    识别显著的因果边，并计算强度和多种置信度指标（内存优化版本）
    
    Args:
        jacobian_mean: 平均 Jacobian 矩阵
        jacobian_std: Jacobian 标准差矩阵
        n_samples: 样本数
        feature_names: 特征名列表
        threshold_percentile: 显著性阈值（百分位数）
        direction: 边的方向标签
    
    Returns:
        edges_df: 显著边的 DataFrame，包含 strength 和多种 confidence 列
    """
    print(f"\n  识别显著边 (阈值: {threshold_percentile}th percentile)...")
    
    # 计算阈值
    abs_jacobian = np.abs(jacobian_mean)
    threshold = np.percentile(abs_jacobian, threshold_percentile)
    
    print(f"    阈值: {threshold:.6f}")
    
    # 找显著边
    significant_mask = abs_jacobian > threshold
    n_edges = significant_mask.sum()
    
    print(f"    显著边数: {n_edges}")
    
    # ========================================================================
    # 置信度指标 1: 跨样本标准差倒数（稳定性）
    # ========================================================================
    print(f"    计算置信度指标 1: 稳定性（标准差倒数）...")
    # 避免除零，使用平滑因子
    stability_matrix = 1.0 / (1.0 + jacobian_std / (abs_jacobian + 1e-8))
    
    # ========================================================================
    # 置信度指标 2: 信噪比（SNR）
    # ========================================================================
    print(f"    计算置信度指标 2: 信噪比（SNR）...")
    snr_matrix = abs_jacobian / (jacobian_std + 1e-8)
    # 归一化到 [0, 1]
    snr_max = snr_matrix.max()
    if snr_max > 0:
        snr_normalized = snr_matrix / snr_max
    else:
        snr_normalized = np.zeros_like(snr_matrix)
    
    # ========================================================================
    # 置信度指标 3: 一致性比例（多少样本支持）- 使用正态分布近似
    # ========================================================================
    print(f"    计算置信度指标 3: 一致性比例（正态分布近似）...")
    # 假设Jacobian值服从正态分布 N(mean, std)
    # 计算有多少比例的样本会超过全局阈值
    from scipy import stats
    global_threshold = threshold  # 使用相同的阈值
    # 对每个元素，计算 P(|X| > threshold)，其中 X ~ N(mean, std)
    # P(|X| > t) = P(X > t) + P(X < -t) = 1 - Φ((t-μ)/σ) + Φ((-t-μ)/σ)
    consistency_matrix = 1 - stats.norm.cdf(
        (global_threshold - abs_jacobian) / (jacobian_std + 1e-8)
    )
    
    print(f"    置信度指标计算完成")
    print(f"      稳定性均值: {stability_matrix[significant_mask].mean():.4f}")
    print(f"      SNR均值: {snr_normalized[significant_mask].mean():.4f}")
    print(f"      一致性均值: {consistency_matrix[significant_mask].mean():.4f}")
    
    # 构建边列表
    row_idx, col_idx = np.where(significant_mask)
    if direction == 'cross_tissue':
        non_self = row_idx != col_idx
        row_idx = row_idx[non_self]
        col_idx = col_idx[non_self]

    edges_df = pd.DataFrame({
        'source': [feature_names[j] for j in col_idx],
        'target': [feature_names[i] for i in row_idx],
        'weight': jacobian_mean[row_idx, col_idx].astype(np.float32),
        'strength': abs_jacobian[row_idx, col_idx].astype(np.float32),
        'confidence_stability': stability_matrix[row_idx, col_idx].astype(np.float32),
        'confidence_snr': snr_normalized[row_idx, col_idx].astype(np.float32),
        'confidence_consistency': consistency_matrix[row_idx, col_idx].astype(np.float32),
        'direction': direction,
    })
    
    # 按强度排序
    edges_df = edges_df.sort_values('strength', ascending=False).reset_index(drop=True)
    
    return edges_df


def main():
    print("="*60)
    print("Step 2: 提取跨组织因果边")
    print("="*60)
    
    for omics_type in ['proteomics','metabolomics','transcriptomics']:
        print(f"\n{'='*60}")
        print(f"处理: {omics_type.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. 加载模型和数据
            model, X_plasma, X_csf, feature_names, cfg = load_model_and_data(omics_type)
            
            # 2. 计算跨组织 Jacobian (反向)
            jacobian_cross_mean, jacobian_cross_std, n_samples_csf = compute_jacobian_brain_to_blood(model, X_csf, cfg)
            edges_cross = identify_significant_edges(jacobian_cross_mean, jacobian_cross_std, n_samples_csf, feature_names, direction='cross_tissue')
            
            # 3. 计算单组织内部 Jacobian
            jacobian_blood_mean, jacobian_blood_std, n_samples_blood = compute_intra_tissue_jacobian(model, X_plasma, cfg, 'blood')
            edges_blood = identify_significant_edges(jacobian_blood_mean, jacobian_blood_std, n_samples_blood, feature_names, direction='intra_blood')
            
            jacobian_brain_mean, jacobian_brain_std, n_samples_brain = compute_intra_tissue_jacobian(model, X_csf, cfg, 'brain')
            edges_brain = identify_significant_edges(jacobian_brain_mean, jacobian_brain_std, n_samples_brain, feature_names, direction='intra_brain')
            
            # 4. 保存结果
            output_dir = OUTPUT_DIR / omics_type
            output_dir.mkdir(exist_ok=True)
            
            # 保存跨组织边
            edges_cross.to_csv(output_dir / 'cross_tissue_edges.csv', index=False)
            np.save(output_dir / 'jacobian_brain_to_blood.npy', jacobian_cross_mean)
            
            # 保存血液内部网络
            blood_dir = output_dir / 'blood_network'
            blood_dir.mkdir(exist_ok=True)
            edges_blood.to_csv(blood_dir / 'consensus_edges.csv', index=False)
            np.save(blood_dir / 'jacobian.npy', jacobian_blood_mean)
            
            # 保存脑组织内部网络
            brain_dir = output_dir / 'brain_network'
            brain_dir.mkdir(exist_ok=True)
            edges_brain.to_csv(brain_dir / 'consensus_edges.csv', index=False)
            np.save(brain_dir / 'jacobian.npy', jacobian_brain_mean)
            
            # 保存统计信息
            stats = {
                'n_features': len(feature_names),
                'n_samples': {
                    'plasma': X_plasma.shape[0],
                    'csf': X_csf.shape[0],
                },
                'cross_tissue': {
                    'n_edges': len(edges_cross),
                    'edge_density': len(edges_cross) / (len(feature_names) ** 2),
                    'mean_strength': float(edges_cross['strength'].mean()),
                    'mean_confidence_stability': float(edges_cross['confidence_stability'].mean()),
                    'mean_confidence_snr': float(edges_cross['confidence_snr'].mean()),
                    'mean_confidence_consistency': float(edges_cross['confidence_consistency'].mean()),
                },
                'blood_network': {
                    'n_edges': len(edges_blood),
                    'edge_density': len(edges_blood) / (len(feature_names) ** 2),
                    'mean_strength': float(edges_blood['strength'].mean()),
                    'mean_confidence_stability': float(edges_blood['confidence_stability'].mean()),
                    'mean_confidence_snr': float(edges_blood['confidence_snr'].mean()),
                    'mean_confidence_consistency': float(edges_blood['confidence_consistency'].mean()),
                },
                'brain_network': {
                    'n_edges': len(edges_brain),
                    'edge_density': len(edges_brain) / (len(feature_names) ** 2),
                    'mean_strength': float(edges_brain['strength'].mean()),
                    'mean_confidence_stability': float(edges_brain['confidence_stability'].mean()),
                    'mean_confidence_snr': float(edges_brain['confidence_snr'].mean()),
                    'mean_confidence_consistency': float(edges_brain['confidence_consistency'].mean()),
                },
            }
            
            with open(output_dir / 'stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\n  ✓ 结果保存在: {output_dir}")
            print(f"    跨组织边: {len(edges_cross)} (密度 {stats['cross_tissue']['edge_density']:.4f})")
            print(f"      置信度 - 稳定性: {stats['cross_tissue']['mean_confidence_stability']:.3f}")
            print(f"      置信度 - SNR: {stats['cross_tissue']['mean_confidence_snr']:.3f}")
            print(f"      置信度 - 一致性: {stats['cross_tissue']['mean_confidence_consistency']:.3f}")
            print(f"    血液网络: {len(edges_blood)} (密度 {stats['blood_network']['edge_density']:.4f})")
            print(f"    脑网络: {len(edges_brain)} (密度 {stats['brain_network']['edge_density']:.4f})")
            
        except Exception as e:
            print(f"\n  ✗ {omics_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Step 2 完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
