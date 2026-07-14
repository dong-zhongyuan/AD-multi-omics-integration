#!/usr/bin/env python3
"""
转录组边预筛选：用血端基因表达特征收窄靶点列表
=================================================
策略：血端 mean_expr Top-200 ∩ var_expr Top-200（交集）
依据：预实验分析表明该交集使生存显著比例从 18.4% → 29%，
      有药物基因从 2.7% → 7%，是所有测试指标中富集效果最好的。

输入：
  - Step3 edges_before_elbow_filtering.csv（全部转录组跨组织边）
  - processed-data/transcriptomics_blood.h5ad（血端表达矩阵）

输出：
  - prescreened_cross_tissue_edges.csv（预筛选后的边）
  - prescreened_blood_genes.csv（血端 KO 候选）
  - prescreened_brain_genes.csv（脑端 KO 候选）

用法：python prescreen_edges.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 参数
TOP_N = 200  # mean_expr 和 var_expr 各取 Top-200 求交集

def main():
    print("=" * 60)
    print("转录组边预筛选")
    print(f"策略：血端 mean_expr Top-{TOP_N} ∩ var_expr Top-{TOP_N}")
    print("=" * 60)

    # 1. 加载 Step3 全部边
    edges_file = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis/transcriptomics/edges_before_elbow_filtering.csv"
    edges = pd.read_csv(edges_file)
    print(f"\n原始边: {len(edges)}")
    print(f"脑端基因: {edges['source'].nunique()}, 血端基因: {edges['target'].nunique()}")

    # 2. 计算血端基因的 mean_expr 和 var_expr
    print("\n计算血端基因表达统计...")
    blood_ad = ad.read_h5ad(PROJECT_ROOT / "processed-data/transcriptomics_blood.h5ad")
    X = blood_ad.X
    if sp.issparse(X):
        mean_expr = np.array(X.mean(axis=0)).flatten()
        var_expr = np.sqrt(np.maximum(
            np.array(X.multiply(X).mean(axis=0)).flatten() - mean_expr**2, 0
        ))
    else:
        mean_expr = X.mean(axis=0)
        var_expr = X.std(axis=0)

    blood_stats = pd.DataFrame({
        'gene': list(blood_ad.var.index.astype(str)),
        'mean_expr': mean_expr,
        'var_expr': var_expr,
    })
    del blood_ad, X

    # 3. 限制在边涉及的血端基因内，取交集
    blood_in_edges = blood_stats[blood_stats['gene'].isin(set(edges['target'].unique()))]
    top_mean = set(blood_in_edges.nlargest(TOP_N, 'mean_expr')['gene'])
    top_var = set(blood_in_edges.nlargest(TOP_N, 'var_expr')['gene'])
    blood_selected = top_mean & top_var

    print(f"\nmean_expr Top-{TOP_N}: {len(top_mean)} 基因")
    print(f"var_expr Top-{TOP_N}: {len(top_var)} 基因")
    print(f"交集: {len(blood_selected)} 基因")

    # 4. 筛选边
    filtered = edges[edges['target'].isin(blood_selected)].copy()
    brain_kept = sorted(set(filtered['source'].unique()))
    print(f"\n预筛选后: {len(filtered)} 边")
    print(f"脑端基因: {len(brain_kept)}")
    print(f"血端基因: {filtered['target'].nunique()}")

    # 5. 保存
    out_dir = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis/transcriptomics"
    filtered.to_csv(out_dir / "prescreened_cross_tissue_edges.csv", index=False)
    pd.DataFrame({'gene': sorted(blood_selected)}).to_csv(out_dir / "prescreened_blood_genes.csv", index=False)
    pd.DataFrame({'gene': brain_kept}).to_csv(out_dir / "prescreened_brain_genes.csv", index=False)

    print(f"\n✅ 已保存:")
    print(f"  {out_dir}/prescreened_cross_tissue_edges.csv")
    print(f"  {out_dir}/prescreened_blood_genes.csv")
    print(f"  {out_dir}/prescreened_brain_genes.csv")


if __name__ == '__main__':
    main()
