#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0: 5xFAD 小鼠单细胞数据预处理（发现队列 + VK 队列统一）

数据来源：GSE329430 (5xFAD vs WT, Brain + Bone Marrow)
- 脑端：Brain_1 (955 WT_1, 956 5xFAD_1, 957 WT_2, 958 5xFAD_2)
- 血端：BM3 + BM4 (骨髓代理外周血)
- 基因映射：小鼠 symbol → 人同源 symbol (MGI 1:1 ortholog)

产出两套文件：
1. 发现队列（Step1 训练用）：
   - transcriptomics_brain.h5ad（脑端, HVG 子集）
   - transcriptomics_blood.h5ad（血端, HVG 子集）
2. VK 队列（Step4 GenKI/Geneformer 用）：
   - step4_single_cell_5xfad/5xFAD_expression_matrix_for_step4.h5ad（脑+血合并, 宽基因集）
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "single-cell" / "GSE329430"
ORTHO_CSV = PROJECT_ROOT / "data" / "metadata" / "orthologs_mgi_1to1.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed-data"
OUTPUT_DIR.mkdir(exist_ok=True)

BRAIN_FILES = [
    (DATA_DIR / "GSM9703846_955_sample_filtered_feature_bc_matrix.h5", "WT", "955"),
    (DATA_DIR / "GSM9703846_956_sample_filtered_feature_bc_matrix.h5", "5xFAD", "956"),
    (DATA_DIR / "GSM9703846_957_sample_filtered_feature_bc_matrix.h5", "WT", "957"),
    (DATA_DIR / "GSM9703846_958_sample_filtered_feature_bc_matrix.h5", "5xFAD", "958"),
]
BM_FILES = [
    (DATA_DIR / "GSM9703842_BM3_filtered_feature_bc_matrix.h5", "BM3"),
    (DATA_DIR / "GSM9703843_BM4_filtered_feature_bc_matrix.h5", "BM4"),
]

# 发现队列参数
MIN_CELLS_PER_GENE = 10
N_TOP_GENES = 2000
RNG = np.random.default_rng(42)

# VK 队列参数
VK_MAX_CELLS_PER_TISSUE = 5000
VK_MIN_CELLS_GLOBAL = 10


def load_ortholog_map() -> dict[str, str]:
    df = pd.read_csv(ORTHO_CSV)
    return dict(zip(df["mouse_symbol"], df["human_symbol"]))


def map_mouse_to_human(adata: ad.AnnData, ortho_map: dict[str, str]) -> ad.AnnData:
    var_names = adata.var_names.astype(str)
    human_syms = [ortho_map.get(g, None) for g in var_names]
    mapped_mask = np.array([h is not None for h in human_syms])
    adata = adata[:, mapped_mask].copy()
    human_syms = [ortho_map[g] for g in var_names[mapped_mask]]
    seen = set()
    keep_idx = []
    for i, h in enumerate(human_syms):
        if h not in seen:
            seen.add(h)
            keep_idx.append(i)
    adata = adata[:, keep_idx].copy()
    adata.var_names = pd.Index([human_syms[i] for i in keep_idx])
    adata.var["mouse_symbol"] = [var_names[mapped_mask][i] for i in keep_idx]
    adata.var_names_make_unique()
    return adata


def preprocess_tissue(adata: ad.AnnData, tissue_name: str) -> ad.AnnData:
    print(f"  [PREP] {tissue_name}: {adata.n_obs} cells × {adata.n_vars} genes")
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"  过滤后: {adata.n_obs} cells × {adata.n_vars} genes")
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata, base=2)
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    return adata


def compute_top_hvgs(brain_common: ad.AnnData, blood_common: ad.AnnData, n_top: int = N_TOP_GENES):
    combined = sp.vstack([brain_common.X, blood_common.X]).tocsr()
    mean = np.asarray(combined.mean(axis=0)).ravel()
    mean_sq = np.asarray(combined.power(2).mean(axis=0)).ravel()
    variance = mean_sq - mean**2
    top_k = min(n_top, len(variance))
    top_idx = np.argsort(variance)[-top_k:]
    top_idx = top_idx[np.argsort(variance[top_idx])[::-1]]
    return brain_common.var_names[top_idx].tolist()


def main():
    print("=" * 60)
    print("Step 0: 5xFAD 预处理（发现队列 + VK 队列统一）")
    print("=" * 60)

    ortho_map = load_ortholog_map()
    print(f"[ORTHO] {len(ortho_map):,} 对同源基因")

    # ================================================================
    # 1. 加载脑端
    # ================================================================
    print("\n[1] 加载脑端（5xFAD vs WT）")
    brain_adatas = []
    for h5_path, genotype, sample_id in BRAIN_FILES:
        print(f"  {h5_path.name} ({genotype})")
        adata = sc.read_10x_h5(str(h5_path))
        adata.var_names_make_unique()
        adata.obs["tissue"] = "Brain"
        adata.obs["genotype"] = genotype
        adata.obs["sample"] = sample_id
        brain_adatas.append(adata)
    adata_brain = ad.concat(brain_adatas, join="inner", merge="same")
    adata_brain.obs_names_make_unique()
    print(f"  脑端合并: {adata_brain.n_obs} × {adata_brain.n_vars}")
    print(f"  基因型: {adata_brain.obs['genotype'].value_counts().to_dict()}")

    # ================================================================
    # 2. 加载血端
    # ================================================================
    print("\n[2] 加载血端（骨髓 BM）")
    blood_adatas = []
    for h5_path, lane in BM_FILES:
        print(f"  {h5_path.name} ({lane})")
        adata = sc.read_10x_h5(str(h5_path))
        adata.var_names_make_unique()
        adata.obs["tissue"] = "Blood"
        adata.obs["genotype"] = "mixed"
        adata.obs["sample"] = lane
        blood_adatas.append(adata)
    adata_blood = ad.concat(blood_adatas, join="inner", merge="same")
    adata_blood.obs_names_make_unique()
    print(f"  血端合并: {adata_blood.n_obs} × {adata_blood.n_vars}")

    # ================================================================
    # 3. 同源映射
    # ================================================================
    print("\n[3] 小鼠 → 人同源基因映射")
    adata_brain = map_mouse_to_human(adata_brain, ortho_map)
    print(f"  脑端: {adata_brain.n_obs} × {adata_brain.n_vars}")
    adata_blood = map_mouse_to_human(adata_blood, ortho_map)
    print(f"  血端: {adata_blood.n_obs} × {adata_blood.n_vars}")

    # 保存原始映射后的数据（给 VK 用）
    brain_full = adata_brain.copy()
    blood_full = adata_blood.copy()

    # ================================================================
    # 4. 发现队列：标准化 + HVG
    # ================================================================
    print("\n[4] 发现队列标准化 + HVG 筛选")
    adata_brain = preprocess_tissue(adata_brain, "Brain")
    adata_blood = preprocess_tissue(adata_blood, "Blood")

    common_genes = sorted(set(adata_brain.var_names) & set(adata_blood.var_names))
    print(f"  [COMMON] {len(common_genes)} 共同基因")
    with open(OUTPUT_DIR / "common_genes_transcriptomics.txt", "w") as f:
        for g in common_genes:
            f.write(f"{g}\n")

    adata_brain_c = adata_brain[:, common_genes].copy()
    adata_blood_c = adata_blood[:, common_genes].copy()

    hvg_genes = compute_top_hvgs(adata_brain_c, adata_blood_c)
    print(f"  [HVG] {len(hvg_genes)} 个高变基因")
    with open(OUTPUT_DIR / "hvg_genes_transcriptomics.txt", "w") as f:
        for g in hvg_genes:
            f.write(f"{g}\n")

    adata_brain_main = adata_brain_c[:, hvg_genes].copy()
    adata_blood_main = adata_blood_c[:, hvg_genes].copy()

    out_brain = OUTPUT_DIR / "transcriptomics_brain.h5ad"
    out_blood = OUTPUT_DIR / "transcriptomics_blood.h5ad"
    adata_brain_main.write(out_brain)
    adata_blood_main.write(out_blood)
    print(f"\n  [SAVE] 发现队列:")
    print(f"    {out_brain.name}: {adata_brain_main.n_obs} × {adata_brain_main.n_vars}")
    print(f"    {out_blood.name}: {adata_blood_main.n_obs} × {adata_blood_main.n_vars}")

    # 样本元数据
    meta_cols = [c for c in ["tissue", "genotype", "sample", "age", "sex"] if c in adata_brain_main.obs.columns]
    all_meta = pd.concat([adata_brain_main.obs[meta_cols], adata_blood_main.obs[meta_cols]])
    all_meta["diagnosis_group"] = all_meta["genotype"].map({"WT": "control", "5xFAD": "case"}).fillna("control")
    all_meta["diagnosis_numeric"] = all_meta["diagnosis_group"].map({"control": 1.0, "case": 3.0})
    all_meta.to_csv(OUTPUT_DIR / "transcriptomics_sample_diagnosis.tsv", sep="\t")

    # ================================================================
    # 5. VK 队列：宽基因集合并 h5ad
    # ================================================================
    print("\n[5] VK 队列构建（宽基因集）")
    brain_vk = brain_full.copy()
    blood_vk = blood_full.copy()

    # VK 标准化
    for adata in [brain_vk, blood_vk]:
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata, base=2)
        adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    # 下采样
    for adata in [brain_vk, blood_vk]:
        if adata.n_obs > VK_MAX_CELLS_PER_TISSUE:
            idx = RNG.choice(adata.n_obs, VK_MAX_CELLS_PER_TISSUE, replace=False)
            adata._inplace_subset_obs(sorted(idx))
    print(f"  VK 下采样: 脑={brain_vk.n_obs}, 血={blood_vk.n_obs}")

    # 合并
    vk_common = sorted(set(brain_vk.var_names) & set(blood_vk.var_names))
    print(f"  VK 共同基因: {len(vk_common)}")
    combined_vk = ad.concat([brain_vk[:, vk_common], blood_vk[:, vk_common]], join="inner", merge="same")
    combined_vk.obs_names_make_unique()
    sc.pp.filter_genes(combined_vk, min_cells=VK_MIN_CELLS_GLOBAL)

    vk_dir = OUTPUT_DIR / "step4_single_cell_5xfad"
    vk_dir.mkdir(exist_ok=True)
    vk_h5ad = vk_dir / "5xFAD_expression_matrix_for_step4.h5ad"
    combined_vk.write_h5ad(vk_h5ad)
    print(f"  [SAVE] VK 队列:")
    print(f"    {vk_h5ad.name}: {combined_vk.n_obs} × {combined_vk.n_vars}")
    print(f"    tissue: {combined_vk.obs['tissue'].value_counts().to_dict()}")

    combined_vk.obs[[c for c in ["tissue", "genotype", "sample"] if c in combined_vk.obs.columns]].to_csv(
        vk_dir / "5xFAD_step4_sample_metadata.tsv", sep="\t"
    )

    print("\n" + "=" * 60)
    print("✅ 5xFAD 预处理完成（发现队列 + VK 队列统一）")
    print("=" * 60)
    print(f"发现队列: Brain {adata_brain_main.n_obs} + Blood {adata_blood_main.n_obs} cells, {len(hvg_genes)} HVG")
    print(f"VK 队列: {combined_vk.n_obs} cells × {combined_vk.n_vars} genes")
    print(f"基因型: {adata_brain.obs['genotype'].value_counts().to_dict()}")
    print(f"数据来源: GSE329430 (5xFAD vs WT)")


if __name__ == "__main__":
    main()
