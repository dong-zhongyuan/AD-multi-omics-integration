import os
#!/usr/bin/env python3
"""
Step 0: 转录组数据预处理（GSE292141 单细胞数据）

目标：
1. 使用 GSE292141 的 PBMC / CSF 单细胞数据作为新的转录组来源
2. 生成与项目现有 step1-step3 文件名兼容的 processed-data 产物
3. 保留样本级诊断信息，供 step3 的疾病相关性分析直接使用

说明：
- GSE292141 的样本标题带有 High/Unknown/Low MOCA 分组
- 这里将其编码为：
    High MOCA   -> diagnosis_numeric = 1.0
    Unknown MOCA -> diagnosis_numeric = 2.0
    Low MOCA    -> diagnosis_numeric = 3.0
- 为控制规模，按样本下采样，每个样本最多保留 2000 个细胞
- 主输出的 transcriptomics_brain.h5ad / transcriptomics_blood.h5ad
  保存的是共同基因上的高变基因子集，便于 step1-step3 直接读取
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

import gzip
import pickle
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from tools.config_loader import get_config

config = get_config()

DATA_DIR = config.get_path("paths.data_dir")
OUTPUT_DIR = config.get_path("paths.processed_data_dir")
OUTPUT_DIR.mkdir(exist_ok=True)

GSE292141_DIR = DATA_DIR / "single-cell" / "GSE292141"
SERIES_MATRIX = GSE292141_DIR / "GSE292141_series_matrix.txt.gz"
MAX_CELLS_PER_SAMPLE = 2000
MIN_CELLS_PER_GENE = 10
N_TOP_GENES = 2000
RNG = np.random.default_rng(42)


def strip_quotes(values):
    return [v.strip().strip('"') for v in values]


def parse_moca_group(title: str) -> str:
    if "High MOCA" in title:
        return "High MOCA"
    if "Low MOCA" in title:
        return "Low MOCA"
    if "Unknown MOCA" in title:
        return "Unknown MOCA"
    raise ValueError(f"无法从标题解析 MOCA 分组: {title}")


def diagnosis_fields_from_moca(moca_group: str):
    if moca_group == "High MOCA":
        return "control", 1.0
    if moca_group == "Unknown MOCA":
        return "unknown", 2.0
    if moca_group == "Low MOCA":
        return "case", 3.0
    raise ValueError(f"未知 MOCA 分组: {moca_group}")


def parse_series_metadata(series_matrix_path: Path) -> pd.DataFrame:
    rows = {}
    with gzip.open(series_matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_title\t"):
                rows["title"] = strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_geo_accession\t"):
                rows["gsm_id"] = strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_source_name_ch1\t"):
                rows["source_name"] = strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_description\t") and "Library name:" in line:
                rows["description"] = strip_quotes(line.rstrip("\n").split("\t")[1:])

    required = {"title", "gsm_id", "source_name", "description"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"series matrix 缺少字段: {sorted(missing)}")

    file_map = {}
    for matrix_file in sorted(GSE292141_DIR.glob("*_matrix.mtx.gz")):
        stem = matrix_file.name.replace("_matrix.mtx.gz", "")
        gsm_id, raw_sample_id = stem.split("_", 1)
        file_map[gsm_id] = raw_sample_id

    records = []
    for title, gsm_id, source_name, description in zip(
        rows["title"],
        rows["gsm_id"],
        rows["source_name"],
        rows["description"],
    ):
        match = re.search(r"Patient\s+(\d+)", title)
        if not match:
            raise ValueError(f"无法从标题解析 patient: {title}")

        patient_num = int(match.group(1))
        tissue = source_name.upper()
        moca_group = parse_moca_group(title)
        diagnosis_group, diagnosis_numeric = diagnosis_fields_from_moca(moca_group)
        sample_id = f"P{patient_num:02d}_{tissue}"

        records.append(
            {
                "sample_id": sample_id,
                "raw_sample_id": file_map[gsm_id],
                "gsm_id": gsm_id,
                "title": title,
                "description": description,
                "tissue": tissue,
                "patient": f"P{patient_num:02d}",
                "moca_group": moca_group,
                "diagnosis": moca_group,
                "diagnosis_group": diagnosis_group,
                "diagnosis_numeric": diagnosis_numeric,
                "dataset": "GSE292141",
            }
        )

    meta = pd.DataFrame(records)
    meta = meta.sort_values(["patient", "tissue"]).reset_index(drop=True)
    return meta


def read_gz_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def load_single_sample(sample_row: pd.Series) -> ad.AnnData:
    raw_sample_id = sample_row["raw_sample_id"]
    gsm_id = sample_row["gsm_id"]
    prefix = f"{gsm_id}_{raw_sample_id}"

    matrix_path = GSE292141_DIR / f"{prefix}_matrix.mtx.gz"
    features_path = GSE292141_DIR / f"{prefix}_features.tsv.gz"
    barcodes_path = GSE292141_DIR / f"{prefix}_barcodes.tsv.gz"

    if not matrix_path.exists() or not features_path.exists() or not barcodes_path.exists():
        raise FileNotFoundError(f"缺少样本文件: {prefix}")

    features = pd.read_csv(
        features_path,
        sep="\t",
        header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    gene_mask = (
        features["feature_type"].eq("Gene Expression")
        & features["gene_id"].astype(str).str.startswith("ENSG")
    )
    features = features.loc[gene_mask].reset_index(drop=True)

    barcodes = read_gz_lines(barcodes_path)
    with gzip.open(matrix_path, "rb") as f:
        matrix = scipy.io.mmread(f).tocsr().T

    matrix = matrix[:, gene_mask.to_numpy()]

    if matrix.shape[0] > MAX_CELLS_PER_SAMPLE:
        keep_idx = np.sort(RNG.choice(matrix.shape[0], MAX_CELLS_PER_SAMPLE, replace=False))
        matrix = matrix[keep_idx]
        barcodes = [barcodes[i] for i in keep_idx]

    obs = pd.DataFrame(index=[f"{sample_row['sample_id']}_{bc}" for bc in barcodes])
    for col in [
        "sample_id",
        "raw_sample_id",
        "gsm_id",
        "title",
        "tissue",
        "patient",
        "moca_group",
        "diagnosis",
        "diagnosis_group",
        "diagnosis_numeric",
        "dataset",
    ]:
        obs[col] = sample_row[col]

    var = pd.DataFrame(index=features["gene_id"].astype(str).tolist())
    var["gene_name"] = features["gene_name"].astype(str).tolist()
    var["feature_type"] = features["feature_type"].astype(str).tolist()

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    return adata


def preprocess_tissue(adata: ad.AnnData, tissue_name: str) -> ad.AnnData:
    print(f"\n[PREP] {tissue_name}: 原始 {adata.n_obs} 细胞 × {adata.n_vars} 基因")
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"[PREP] {tissue_name}: 过滤后 {adata.n_obs} 细胞 × {adata.n_vars} 基因")
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata, base=2)
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    return adata


def compute_top_hvgs(brain_common: ad.AnnData, blood_common: ad.AnnData):
    combined = sp.vstack([brain_common.X, blood_common.X]).tocsr()
    mean = np.asarray(combined.mean(axis=0)).ravel()
    mean_sq = np.asarray(combined.power(2).mean(axis=0)).ravel()
    variance = mean_sq - mean**2
    top_k = min(N_TOP_GENES, len(variance))
    top_idx = np.argsort(variance)[-top_k:]
    top_idx = top_idx[np.argsort(variance[top_idx])[::-1]]
    genes = brain_common.var_names[top_idx].tolist()
    stats = pd.DataFrame(
        {
            "gene_id": brain_common.var_names.tolist(),
            "variance": variance,
        }
    ).sort_values("variance", ascending=False)
    return genes, stats


def mean_by_sample(adata: ad.AnnData) -> pd.DataFrame:
    rows = []
    for sample_id in adata.obs["sample_id"].astype(str).unique():
        mask = adata.obs["sample_id"].astype(str).values == sample_id
        sample_mat = adata.X[mask]
        sample_mean = np.asarray(sample_mat.mean(axis=0)).ravel()
        row = {"sample_id": sample_id}
        row.update({gene: sample_mean[i] for i, gene in enumerate(adata.var_names)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_deg_ranking(brain_hvg: ad.AnnData, blood_hvg: ad.AnnData, metadata: pd.DataFrame) -> pd.DataFrame:
    brain_means = mean_by_sample(brain_hvg)
    blood_means = mean_by_sample(blood_hvg)

    dx_cols = ["sample_id", "diagnosis", "diagnosis_group", "diagnosis_numeric"]
    brain_means = brain_means.merge(metadata[metadata["tissue"] == "CSF"][dx_cols], on="sample_id", how="left")
    blood_means = blood_means.merge(metadata[metadata["tissue"] == "PBMC"][dx_cols], on="sample_id", how="left")

    control_mask_brain = brain_means["diagnosis_group"] == "control"
    case_mask_brain = brain_means["diagnosis_group"] == "case"
    control_mask_blood = blood_means["diagnosis_group"] == "control"
    case_mask_blood = blood_means["diagnosis_group"] == "case"

    genes = brain_hvg.var_names.tolist()
    brain_control = brain_means.loc[control_mask_brain, genes].mean(axis=0)
    brain_case = brain_means.loc[case_mask_brain, genes].mean(axis=0)
    blood_control = blood_means.loc[control_mask_blood, genes].mean(axis=0)
    blood_case = blood_means.loc[case_mask_blood, genes].mean(axis=0)

    deg = pd.DataFrame(
        {
            "gene_id": genes,
            "brain_control_mean": brain_control.values,
            "brain_case_mean": brain_case.values,
            "blood_control_mean": blood_control.values,
            "blood_case_mean": blood_case.values,
        }
    )
    deg["brain_logfc"] = deg["brain_case_mean"] - deg["brain_control_mean"]
    deg["blood_logfc"] = deg["blood_case_mean"] - deg["blood_control_mean"]
    deg["combined_abs_logfc"] = deg["brain_logfc"].abs() + deg["blood_logfc"].abs()
    deg = deg.sort_values(["combined_abs_logfc", "gene_id"], ascending=[False, True]).reset_index(drop=True)
    return deg


def save_simple_gene_list(path: Path, genes):
    pd.DataFrame({"gene_id": list(genes)}).to_csv(path, index=False)


def main():
    print("=" * 60)
    print("Step 0: 转录组数据预处理 (GSE292141 单细胞数据)")
    print("=" * 60)

    metadata = parse_series_metadata(SERIES_MATRIX)
    metadata_out = DATA_DIR / "metadata" / "GSE292141_processed_sample_diagnosis.tsv"
    metadata.to_csv(metadata_out, sep="\t", index=False)
    print(f"[META] 保存样本诊断表: {metadata_out}")

    brain_adatas = []
    blood_adatas = []

    print("\n[LOAD] 逐样本读取 GSE292141...")
    for _, row in metadata.iterrows():
        adata = load_single_sample(row)
        print(
            f"  {row['sample_id']} ({row['gsm_id']}, {row['moca_group']}): "
            f"{adata.n_obs} 细胞 × {adata.n_vars} 基因"
        )
        if row["tissue"] == "CSF":
            brain_adatas.append(adata)
        else:
            blood_adatas.append(adata)

    if not brain_adatas or not blood_adatas:
        raise RuntimeError("GSE292141 缺少 CSF 或 PBMC 样本，无法继续")

    adata_brain = ad.concat(brain_adatas, join="inner", merge="same")
    adata_blood = ad.concat(blood_adatas, join="inner", merge="same")

    adata_brain = preprocess_tissue(adata_brain, "CSF")
    adata_blood = preprocess_tissue(adata_blood, "PBMC")

    common_genes = sorted(set(adata_brain.var_names) & set(adata_blood.var_names))
    if not common_genes:
        raise RuntimeError("PBMC/CSF 没有共同基因，无法生成下游文件")

    print(f"\n[COMMON] 共同基因数: {len(common_genes)}")
    with open(OUTPUT_DIR / "common_genes_transcriptomics.txt", "w", encoding="utf-8") as f:
        for gene in common_genes:
            f.write(f"{gene}\n")

    adata_brain_common = adata_brain[:, common_genes].copy()
    adata_blood_common = adata_blood[:, common_genes].copy()

    hvg_genes, hvg_stats = compute_top_hvgs(adata_brain_common, adata_blood_common)
    print(f"[HVG] 选择 {len(hvg_genes)} 个高变基因用于主输出")
    with open(OUTPUT_DIR / "hvg_genes_transcriptomics.txt", "w", encoding="utf-8") as f:
        for gene in hvg_genes:
            f.write(f"{gene}\n")

    adata_brain_main = adata_brain_common[:, hvg_genes].copy()
    adata_blood_main = adata_blood_common[:, hvg_genes].copy()

    # 主输出：step1-step3 直接读取
    out_brain = OUTPUT_DIR / "transcriptomics_brain.h5ad"
    out_blood = OUTPUT_DIR / "transcriptomics_blood.h5ad"
    adata_brain_main.write(out_brain)
    adata_blood_main.write(out_blood)
    print(f"[SAVE] {out_brain} -> {adata_brain_main.n_obs} × {adata_brain_main.n_vars}")
    print(f"[SAVE] {out_blood} -> {adata_blood_main.n_obs} × {adata_blood_main.n_vars}")

    # HVG 别名输出
    adata_brain_main.write(OUTPUT_DIR / "transcriptomics_brain_hvg.h5ad")
    adata_blood_main.write(OUTPUT_DIR / "transcriptomics_blood_hvg.h5ad")
    save_simple_gene_list(OUTPUT_DIR / "transcriptomics_brain_hvg.csv", hvg_genes)
    save_simple_gene_list(OUTPUT_DIR / "transcriptomics_blood_hvg.csv", hvg_genes)

    # 生成样本级诊断表（供 step3 直接读取）
    sample_dx = metadata[
        [
            "sample_id",
            "raw_sample_id",
            "gsm_id",
            "title",
            "tissue",
            "patient",
            "moca_group",
            "diagnosis",
            "diagnosis_group",
            "diagnosis_numeric",
            "dataset",
        ]
    ].copy()
    sample_dx.to_csv(OUTPUT_DIR / "transcriptomics_sample_diagnosis.tsv", sep="\t", index=False)
    sample_dx[sample_dx["tissue"] == "CSF"].to_csv(
        OUTPUT_DIR / "transcriptomics_brain_sample_diagnosis.tsv",
        sep="\t",
        index=False,
    )
    sample_dx[sample_dx["tissue"] == "PBMC"].to_csv(
        OUTPUT_DIR / "transcriptomics_blood_sample_diagnosis.tsv",
        sep="\t",
        index=False,
    )

    # 生成 PCA / paired 产物
    n_components = min(500, len(hvg_genes), adata_brain_main.n_obs, adata_blood_main.n_obs)
    print(f"[PCA] TruncatedSVD 维度: {n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_combined = sp.vstack([adata_blood_main.X, adata_brain_main.X]).tocsr()
    svd.fit(X_combined)

    blood_pca = svd.transform(adata_blood_main.X)
    brain_pca = svd.transform(adata_brain_main.X)
    explained = float(svd.explained_variance_ratio_.sum())

    blood_pca_adata = ad.AnnData(
        X=blood_pca,
        obs=adata_blood_main.obs.copy(),
        var=pd.DataFrame(index=[f"PC{i+1}" for i in range(n_components)]),
    )
    brain_pca_adata = ad.AnnData(
        X=brain_pca,
        obs=adata_brain_main.obs.copy(),
        var=pd.DataFrame(index=[f"PC{i+1}" for i in range(n_components)]),
    )
    blood_pca_adata.uns["pca_loadings"] = svd.components_
    brain_pca_adata.uns["pca_loadings"] = svd.components_
    blood_pca_adata.uns["pca_feature_names"] = hvg_genes
    brain_pca_adata.uns["pca_feature_names"] = hvg_genes

    blood_pca_adata.write(OUTPUT_DIR / "transcriptomics_blood_paired.h5ad")
    brain_pca_adata.write(OUTPUT_DIR / "transcriptomics_brain_paired.h5ad")
    with open(OUTPUT_DIR / "transcriptomics_pca_model.pkl", "wb") as f:
        pickle.dump(
            {
                "pca": svd,
                "hvg_genes": hvg_genes,
                "common_genes": common_genes,
                "n_components": n_components,
                "explained_variance_ratio": svd.explained_variance_ratio_,
                "loading_matrix": svd.components_,
            },
            f,
        )
    print(f"[PCA] 解释方差总和: {explained:.2%}")

    # 生成 DEG 兼容产物（基于 High vs Low MOCA 的伪 bulk 排名）
    deg_df = build_deg_ranking(adata_brain_main, adata_blood_main, metadata)
    deg_df.to_csv(OUTPUT_DIR / "transcriptomics_deg_genes.csv", index=False)
    deg_genes = deg_df["gene_id"].tolist()

    adata_brain_deg = adata_brain_main[:, deg_genes].copy()
    adata_blood_deg = adata_blood_main[:, deg_genes].copy()
    adata_brain_deg.write(OUTPUT_DIR / "transcriptomics_brain_deg.h5ad")
    adata_blood_deg.write(OUTPUT_DIR / "transcriptomics_blood_deg.h5ad")
    save_simple_gene_list(OUTPUT_DIR / "transcriptomics_brain_deg.csv", deg_genes)
    save_simple_gene_list(OUTPUT_DIR / "transcriptomics_blood_deg.csv", deg_genes)

    print("\n" + "=" * 60)
    print("✅ GSE292141 转录组预处理完成")
    print("=" * 60)
    print(f"CSF 主输出:  {adata_brain_main.n_obs} 细胞 × {adata_brain_main.n_vars} 基因")
    print(f"PBMC 主输出: {adata_blood_main.n_obs} 细胞 × {adata_blood_main.n_vars} 基因")
    print(f"样本分组: {sample_dx['diagnosis'].value_counts().to_dict()}")
    print(f"数据来源: {GSE292141_DIR}")


if __name__ == "__main__":
    main()
