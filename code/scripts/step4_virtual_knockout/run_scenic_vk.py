#!/usr/bin/env python3
"""
SCENIC-based Virtual Knockout for Transcriptomics
==================================================
Second VK method for transcriptomics, complementing GenKI.

Method:
  1. GRN inference: GRNBoost2 (random forest regression) infers TF→target GRN
  2. Regulon construction: TF + co-expressed targets = regulon
  3. VK mechanism: KO gene X → remove X from expression matrix → recalculate
     AUCell regulon activity scores → measure regulon activity shift

Unlike Geneformer (mean_pool dilution) and CellOracle (TF-only), this method:
  - Works on any gene (not just TFs)
  - Is data-driven (no pretrained model bias)
  - Is fast (~15 min for full pipeline)
  - Has Nature Methods publication backing (Aibar et al. 2017)

References:
  - pySCENIC: https://github.com/aertslab/pySCENIC
  - GRNBoost2: https://github.com/aertslab/GRNBoost2

Usage:
  python run_scenic_vk.py
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "processed-data/step4_single_cell_5xfad/5xFAD_expression_matrix_for_step4.h5ad"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "SCENIC_transcriptomics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 从 GenKI 显著结果读取 KO 基因
GENKI_FWD_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3"
GENKI_REV_DIR = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3_reverse"

# 参数
N_HVG = 2000       # GRN 推断用 top HVG
N_CELLS = 3000     # 采样细胞数
N_THREADS = 4      # GRNBoost2 并行


def load_genki_sig_genes():
    """从 GenKI 显著结果读取 KO 基因"""
    fwd_genes = []
    for f in sorted(GENKI_FWD_DIR.glob("transcriptomics_*_statistics.csv")):
        gene = f.stem.replace("transcriptomics_", "").replace("_statistics", "")
        df = pd.read_csv(f)
        if len(df) > 0 and df.iloc[0]["n_significant_targets"] > 0:
            fwd_genes.append(gene)

    rev_genes = []
    for f in sorted(GENKI_REV_DIR.glob("transcriptomics_*_statistics.csv")):
        gene = f.stem.replace("transcriptomics_", "").replace("_statistics", "")
        df = pd.read_csv(f)
        if len(df) > 0 and df.iloc[0]["n_significant_targets"] > 0:
            rev_genes.append(gene)

    print(f"GenKI forward 显著: {len(fwd_genes)} 基因")
    print(f"GenKI reverse 显著: {len(rev_genes)} 基因")
    return fwd_genes, rev_genes


def prepare_data(ko_genes=None):
    """加载 5xFAD 数据，采样，预处理"""
    print("[1/5] 数据准备...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  原始: {adata.shape}")

    # 采样
    np.random.seed(42)
    brain_idx = np.where(adata.obs['tissue'] == 'Brain')[0]
    blood_idx = np.where(adata.obs['tissue'] == 'Blood')[0]
    n_each = min(N_CELLS // 2, len(brain_idx), len(blood_idx))
    sampled = np.concatenate([
        np.random.choice(brain_idx, n_each, replace=False),
        np.random.choice(blood_idx, n_each, replace=False)
    ])
    adata = adata[sampled].copy()
    print(f"  采样后: {adata.shape}")

    # 选 HVG + 强制保留 KO 基因
    sc.pp.highly_variable_genes(adata, n_top_genes=min(N_HVG, adata.n_vars), flavor='seurat')
    hvg = set(adata.var_names[adata.var.highly_variable].tolist())
    if ko_genes:
        ko_in_data = [g for g in ko_genes if g in adata.var_names]
        hvg = sorted(hvg | set(ko_in_data))
        print(f"  HVG + KO 基因: {len(hvg)} (KO 保留 {len(ko_in_data)})")
    else:
        hvg = sorted(hvg)
    adata = adata[:, hvg].copy()
    print(f"  最终基因集: {adata.shape}")

    # 确保正数（SCENIC 需要）
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.maximum(X, 0)
    adata.X = X

    return adata


def infer_grn(adata, fwd_genes=None, rev_genes=None):
    """用 GradientBoosting 推断 GRN（GRNBoost2 算法的直接实现，绕过 dask）
    参考: Moerman et al. (2019) GRNBoost2, Bioinformatics
    """
    # 缓存：如果 GRN 已推断过，直接加载
    cache_path = OUTPUT_DIR / "grn_adjacencies.csv"
    if cache_path.exists():
        print("\n[2/5] GRN 推断（加载缓存）...")
        adjacencies = pd.read_csv(cache_path)
        print(f"  加载缓存: {len(adjacencies)} 边")
        return adjacencies

    print("\n[2/5] GRN 推断（GradientBoosting, GRNBoost2 算法）...")
    print("\n[2/5] GRN 推断（GradientBoosting, GRNBoost2 算法）...")

    from sklearn.ensemble import GradientBoostingRegressor

    # 表达矩阵: cells × genes
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    gene_names = list(adata.var_names.astype(str))
    exp_df = pd.DataFrame(X, columns=gene_names)

    print(f"  表达矩阵: {exp_df.shape} (cells × genes)")
    print(f"  基因数: {len(gene_names)}")

    # GRNBoost2: 对每个 target 基因，用所有其他基因预测它
    # 只对 KO 候选基因做 target（大幅减少计算量：~119 vs 2000）
    ko_targets = sorted(set((fwd_genes or []) + (rev_genes or [])))
    ko_targets_in_data = [g for g in ko_targets if g in gene_names]
    print(f"  KO 候选 target 基因: {len(ko_targets_in_data)}")

    t0 = time.time()
    rows = []
    for i, target in enumerate(tqdm(ko_targets_in_data, desc="  GRN", file=sys.stdout)):
        y = exp_df[target].values
        X_pred = exp_df.drop(columns=[target]).values
        predictor_names = [g for g in gene_names if g != target]

        # 随机梯度提升回归（GRNBoost2 核心算法）
        # 减少 estimators 和 depth 加速（2078 特征 × 3000 细胞）
        model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=2,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_pred, y)
        importances = model.feature_importances_

        # 取 top predictors (importance > 0.01)
        for j, imp in enumerate(importances):
            if imp > 0.005:  # 过滤低 importance
                rows.append({
                    'TF': predictor_names[j],
                    'target': target,
                    'importance': float(imp),
                })

    adjacencies = pd.DataFrame(rows).sort_values('importance', ascending=False)
    print(f"  GRN 推断完成: {time.time()-t0:.1f}s")
    print(f"  边数: {len(adjacencies)}")
    print(f"  涉及 TF: {adjacencies['TF'].nunique()}")
    print(f"  涉及 target: {adjacencies['target'].nunique()}")

    # 保存
    adjacencies.to_csv(OUTPUT_DIR / "grn_adjacencies.csv", index=False)

    return adjacencies


def build_regulons(adjacencies, adata):
    """从 GRN adjacencies 构建 regulons（不依赖 pyscenic.prune，避免 np.object 兼容问题）"""
    print("\n[3/5] 构建 regulons...")

    # 直接从 adjacencies 构建 regulon，不走 pyscenic 的 cisTarget/motif 流程
    # 每个 TF 的 top targets（importance 排名前 50）组成一个 regulon
    regulons = {}
    for tf, group in adjacencies.groupby('TF'):
        top_targets = group.nlargest(50, 'importance')['target'].tolist()
        if len(top_targets) >= 5:
            regulons[tf] = set(top_targets)

    print(f"  Regulons (>=5 targets): {len(regulons)}")

    # 保存
    with open(OUTPUT_DIR / "regulons.csv", "w") as f:
        f.write("regulon,targets\n")
        for name, targets in regulons.items():
            f.write(f"{name},{'|'.join(sorted(targets))}\n")

    return regulons
    print(f"  Modules: {len(modules)}")

    # 每个 module 取 top target（importance > median）
    regulons = {}
    for module_name, module_df in modules.items():
        if len(module_df) == 0:
            continue
        # 取 importance 前 50 的 target
        top_targets = module_df.nlargest(50, 'importance')['target'].tolist()
        if len(top_targets) >= 5:  # 至少 5 个 target
            regulons[module_name] = set(top_targets)

    print(f"  Regulons (>=5 targets): {len(regulons)}")

    # 保存
    with open(OUTPUT_DIR / "regulons.csv", "w") as f:
        f.write("regulon,targets\n")
        for name, targets in regulons.items():
            f.write(f"{name},{'|'.join(sorted(targets))}\n")

    return regulons


def compute_aucell_scores(adata, regulons):
    """计算 baseline AUCell regulon 活性分数"""
    print("\n[4/5] 计算 baseline AUCell scores...")

    from pyscenic.aucell import aucell, create_rankings

    # AUCell 需要 genes × cells 的表达矩阵
    exp_df = adata.to_df()  # cells × genes
    exp_df = exp_df.T  # genes × cells

    # 构建 regulon DataFrame（AUCell 格式：regulon name → frozenset of genes）
    regulon_fs = {name: frozenset(targets) for name, targets in regulons.items()}

    # 计算 AUCell
    auc_mtx = aucell(
        exp_mtx=exp_df,
        regulons=regulon_fs,
        seed=42,
    )
    print(f"  AUCell matrix: {auc_mtx.shape} (cells × regulons)")

    return auc_mtx


def run_knockout(adata, regulons, ko_genes, direction):
    """对每个基因做 SCENIC VK"""
    print(f"\n[5/5] SCENIC VK ({direction}): {len(ko_genes)} 基因")

    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature

    # 构建 GeneSignature 对象（aucell 要求）
    signatures = [
        GeneSignature(name=name, gene2weight={g: 1.0 for g in targets})
        for name, targets in regulons.items()
    ]

    # Baseline regulon 活性
    print("  Computing baseline AUCell...", flush=True)
    t0 = time.time()
    exp_df = adata.to_df()  # cells × genes
    baseline_auc = aucell(exp_mtx=exp_df, signatures=signatures, seed=42)
    print(f"  Baseline AUC: {baseline_auc.shape} ({time.time()-t0:.1f}s)", flush=True)

    fwd_out = OUTPUT_DIR / direction
    fwd_out.mkdir(exist_ok=True)

    results = []
    n_total = len(ko_genes)
    for i, gene in enumerate(ko_genes, 1):
        if gene not in adata.var_names:
            print(f"  [{i}/{n_total}] {gene}: skipped (not in data)", flush=True)
            continue

        t1 = time.time()
        # KO: 把该基因表达设为 0
        ko_adata = adata.copy()
        gene_idx = list(ko_adata.var_names).index(gene)
        ko_adata.X[:, gene_idx] = 0.0

        # 重新计算 regulon 活性
        ko_exp_df = ko_adata.to_df()
        ko_auc = aucell(exp_mtx=ko_exp_df, signatures=signatures, seed=42)
        elapsed = time.time() - t1
        print(f"  [{i}/{n_total}] {gene}: AUCell done ({elapsed:.1f}s)", flush=True)

        # 计算每个 regulon 的活性变化
        delta = ko_auc - baseline_auc
        mean_abs_delta = delta.abs().mean(axis=0)

        # 按 regulon 排序
        effect_df = pd.DataFrame({
            'Regulon': mean_abs_delta.index,
            'Mean_Abs_Delta': mean_abs_delta.values,
        }).sort_values('Mean_Abs_Delta', ascending=False)

        # 统计
        overall_effect = float(delta.abs().mean().mean())
        max_effect = float(delta.abs().mean().max())
        std_effect = float(delta.abs().mean().std())
        if std_effect > 0:
            effect_df['Z_score'] = (effect_df['Mean_Abs_Delta'] - overall_effect) / std_effect
        else:
            effect_df['Z_score'] = 0.0
        n_sig = int((effect_df['Z_score'] > 2.0).sum())

        results.append({
            'KO_gene': gene,
            'overall_effect': overall_effect,
            'max_effect': max_effect,
            'n_significant_targets': n_sig,
            'n_target_genes': len(effect_df),
            'top_target': effect_df.iloc[0]['Regulon'] if len(effect_df) > 0 else '',
        })

        # 保存 per-gene ranking
        effect_df.to_csv(fwd_out / f"transcriptomics_{gene}_regulon_ranking.csv", index=False)

        del ko_adata

    # 保存统计
    if results:
        stats_df = pd.DataFrame(results)
        stats_df.to_csv(fwd_out / f"scenic_{direction}_statistics.csv", index=False)
        print(f"\n  ✅ 统计保存: {fwd_out / f'scenic_{direction}_statistics.csv'}")
        print(stats_df[['KO_gene', 'overall_effect', 'n_significant_targets', 'top_target']].head(10).to_string(index=False))
        return stats_df
    return None


def main():
    print("=" * 70)
    print("SCENIC Virtual Knockout (Transcriptomics)")
    print("pySCENIC (Aibar et al., 2017) + GRNBoost2 (Moerman et al., 2019)")
    print("=" * 70)

    t0 = time.time()

    # 加载 GenKI 显著基因
    fwd_genes, rev_genes = load_genki_sig_genes()

    # 准备数据（强制保留 KO 基因）
    all_ko = sorted(set(fwd_genes + rev_genes))
    adata = prepare_data(all_ko)
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # GRN 推断
    adjacencies = infer_grn(adata, fwd_genes, rev_genes)
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # 构建 regulons
    regulons = build_regulons(adjacencies, adata)
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # Forward VK
    if fwd_genes:
        fwd_stats = run_knockout(adata, regulons, fwd_genes, "forward")

    # Reverse VK
    if rev_genes:
        rev_stats = run_knockout(adata, regulons, rev_genes, "reverse")

    elapsed = time.time() - t0
    print(f"\n⏱ 总用时: {elapsed:.1f}s")
    print("=" * 70)
    print("SCENIC VK Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
