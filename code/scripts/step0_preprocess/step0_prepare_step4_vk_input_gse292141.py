import os
#!/usr/bin/env python3
"""
Step 0 (step4 专用): 从 GSE292141 原始单细胞矩阵构建虚拟敲除输入文件

用途：
- 只服务于 step4 虚拟敲除，不替代 step1-step3 使用的主 processed-data 文件
- 直接从 GSE292141 原始 10x 矩阵生成更宽基因集的单细胞输入
- 尽量保留当前 step3 输出里会真正用到的 proteomics / transcriptomics 保护基因

输出：
- processed-data/step4_single_cell_gse292141/GSE292141_expression_matrix_for_step4.h5ad
- processed-data/step4_single_cell_gse292141/GSE292141_step4_sample_metadata.tsv
- processed-data/step4_single_cell_gse292141/GSE292141_step4_protected_gene_coverage.tsv
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config
from tools.gene_id_converter import ensembl_to_symbol

config = get_config()

DATA_DIR = config.get_path("paths.data_dir")
OUTPUT_DIR = config.get_path("paths.processed_data_dir") / "step4_single_cell_gse292141"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = DATA_DIR / "single-cell" / "GSE292141"
SERIES_MATRIX = RAW_DIR / "GSE292141_series_matrix.txt.gz"
OUT_H5AD = OUTPUT_DIR / "GSE292141_expression_matrix_for_step4.h5ad"
OUT_META = OUTPUT_DIR / "GSE292141_step4_sample_metadata.tsv"
OUT_COVERAGE = OUTPUT_DIR / "GSE292141_step4_protected_gene_coverage.tsv"

MIN_CELLS_GLOBAL = 10


def _strip_quotes(values):
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
                rows["title"] = _strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_geo_accession\t"):
                rows["gsm_id"] = _strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_source_name_ch1\t"):
                rows["source_name"] = _strip_quotes(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!Sample_description\t") and "Library name:" in line:
                rows["description"] = _strip_quotes(line.rstrip("\n").split("\t")[1:])

    file_map = {}
    for matrix_file in sorted(RAW_DIR.glob("*_matrix.mtx.gz")):
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
        patient_num = int(title.split()[1])
        tissue = source_name.upper()
        moca_group = parse_moca_group(title)
        diagnosis_group, diagnosis_numeric = diagnosis_fields_from_moca(moca_group)
        records.append(
            {
                "sample_id": f"P{patient_num:02d}_{tissue}",
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

    meta = pd.DataFrame(records).sort_values(["patient", "tissue"]).reset_index(drop=True)
    return meta


def load_protected_genes() -> set[str]:
    """Collect genes that step4 may need from current step3 outputs."""
    protected = set()
    step3_dir = PROJECT_ROOT / "output" / "step3_hub_identification" / "eigengene_analysis"

    # Proteomics edges are already symbol-like after step3 post-processing.
    prot_edges = step3_dir / "proteomics" / "filtered_cross_tissue_edges.csv"
    if prot_edges.exists():
        df = pd.read_csv(prot_edges)
        protected.update(df["source"].dropna().astype(str))
        protected.update(df["target"].dropna().astype(str))

    # Transcriptomics edges still use Ensembl IDs in step3 output; convert them to symbols.
    tx_edges = step3_dir / "transcriptomics" / "filtered_cross_tissue_edges.csv"
    if tx_edges.exists():
        df = pd.read_csv(tx_edges)
        ensembl_ids = list(
            set(df["source"].dropna().astype(str)).union(set(df["target"].dropna().astype(str)))
        )
        mapping = ensembl_to_symbol(ensembl_ids)
        for ensg in ensembl_ids:
            protected.add(mapping.get(ensg, ensg))

    protected = {g for g in protected if g and g != "nan"}
    print(f"[PROTECT] 当前 step3 保护基因数: {len(protected)}")
    return protected


def build_symbol_aggregation(first_feature_path: Path):
    features = pd.read_csv(
        first_feature_path,
        sep="\t",
        header=None,
        names=["ensembl_id", "gene_name", "feature_type"],
    )
    gene_mask = (
        features["feature_type"].eq("Gene Expression")
        & features["gene_name"].notna()
        & features["gene_name"].astype(str).ne("")
    )
    features = features.loc[gene_mask].reset_index(drop=True)

    gene_names = features["gene_name"].astype(str).to_numpy()
    ensembl_ids = features["ensembl_id"].astype(str).to_numpy()
    codes, unique_symbols = pd.factorize(gene_names, sort=False)

    agg = sp.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float32),
            (np.arange(len(codes)), codes),
        ),
        shape=(len(codes), len(unique_symbols)),
    )

    symbol_to_ensembls = (
        features.groupby("gene_name")["ensembl_id"]
        .apply(lambda s: ";".join(pd.unique(s.astype(str))))
        .to_dict()
    )

    return {
        "gene_mask": gene_mask.to_numpy(),
        "unique_symbols": unique_symbols.astype(str).tolist(),
        "agg_matrix": agg,
        "symbol_to_ensembls": symbol_to_ensembls,
    }


def load_sample_matrix(sample_row: pd.Series, agg_info: dict):
    prefix = f"{sample_row['gsm_id']}_{sample_row['raw_sample_id']}"
    matrix_path = RAW_DIR / f"{prefix}_matrix.mtx.gz"
    barcodes_path = RAW_DIR / f"{prefix}_barcodes.tsv.gz"

    with gzip.open(matrix_path, "rb") as f:
        raw = scipy.io.mmread(f).tocsr().T  # cells x features

    raw = raw[:, agg_info["gene_mask"]]
    aggregated = raw @ agg_info["agg_matrix"]  # cells x unique gene symbols
    barcodes = [
        line.rstrip("\n")
        for line in gzip.open(barcodes_path, "rt", encoding="utf-8", errors="ignore")
    ]
    return aggregated.tocsr(), barcodes


def main():
    print("=" * 60)
    print("Step 0 (step4专用): 构建 GSE292141 虚拟敲除输入文件")
    print("=" * 60)

    metadata = parse_series_metadata(SERIES_MATRIX)
    metadata.to_csv(OUT_META, sep="\t", index=False)
    print(f"[META] 保存样本元数据: {OUT_META}")

    protected_genes = load_protected_genes()

    first_feature = sorted(RAW_DIR.glob("*_features.tsv.gz"))[0]
    agg_info = build_symbol_aggregation(first_feature)
    unique_symbols = agg_info["unique_symbols"]
    print(f"[FEATURES] 原始 symbol 数: {len(unique_symbols)}")

    global_nonzero = np.zeros(len(unique_symbols), dtype=np.int64)
    sample_blocks = []
    sample_obs = []

    print("\n[PASS 1] 逐样本聚合 gene symbol 并统计表达覆盖...")
    for _, row in metadata.iterrows():
        mat, barcodes = load_sample_matrix(row, agg_info)
        global_nonzero += np.asarray((mat > 0).sum(axis=0)).ravel().astype(np.int64)
        sample_blocks.append((row.copy(), mat, barcodes))
        print(
            f"  {row['sample_id']} ({row['gsm_id']}): "
            f"{mat.shape[0]} 细胞 × {mat.shape[1]} symbol"
        )

    token_dict = None
    token_path = PROJECT_ROOT / "tools" / "geneformer-main" / "geneformer" / "token_dictionary_gc104M.pkl"
    if token_path.exists():
        import pickle

        with open(token_path, "rb") as f:
            token_dict = pickle.load(f)
    token_genes = set(token_dict.keys()) if token_dict else set()

    keep_mask = np.array(
        [
            (global_nonzero[i] >= MIN_CELLS_GLOBAL) or (gene in protected_genes)
            for i, gene in enumerate(unique_symbols)
        ],
        dtype=bool,
    )
    kept_symbols = [g for g, keep in zip(unique_symbols, keep_mask) if keep]
    print(f"\n[SELECT] 保留基因数: {len(kept_symbols)} / {len(unique_symbols)}")
    print(f"         保护基因总数: {len(protected_genes)}")

    kept_symbol_set = set(kept_symbols)
    coverage = pd.DataFrame(
        {
            "gene": sorted(protected_genes),
        }
    )
    coverage["present_in_step4_input"] = coverage["gene"].isin(kept_symbol_set)
    coverage.to_csv(OUT_COVERAGE, sep="\t", index=False)
    present = int(coverage["present_in_step4_input"].sum())
    print(f"[COVERAGE] 保护基因命中: {present}/{len(coverage)}")
    print(f"           详情: {OUT_COVERAGE}")

    symbol_to_ensembl = agg_info["symbol_to_ensembls"]
    var = pd.DataFrame(index=kept_symbols)
    var["gene_name"] = kept_symbols
    var["ensembl_id"] = [symbol_to_ensembl[g] for g in kept_symbols]
    var["feature_type"] = "Gene Expression"
    var["n_cells"] = global_nonzero[keep_mask]
    var["is_protected_for_step4"] = [g in protected_genes for g in kept_symbols]
    var["in_geneformer_token_dict"] = [g in token_genes for g in kept_symbols]

    print("\n[PASS 2] 构建合并 AnnData...")
    adatas = []
    for row, mat, barcodes in sample_blocks:
        mat_keep = mat[:, keep_mask].tocsr()
        obs_index = [f"{row['sample_id']}_{bc}" for bc in barcodes]
        obs = pd.DataFrame(index=obs_index)
        for col in metadata.columns:
            obs[col] = row[col]
        obs["cell_barcode"] = barcodes
        adatas.append(ad.AnnData(X=mat_keep, obs=obs, var=var.copy()))
        print(f"  add {row['sample_id']}: {mat_keep.shape[0]} cells × {mat_keep.shape[1]} genes")

    combined = ad.concat(adatas, join="inner", merge="same")
    combined.write_h5ad(OUT_H5AD)

    print("\n" + "=" * 60)
    print("✅ Step4 专用输入文件已生成")
    print("=" * 60)
    print(f"输出文件: {OUT_H5AD}")
    print(f"shape: {combined.shape}")
    print(f"tissue counts: {combined.obs['tissue'].value_counts().to_dict()}")
    print(f"sample_id count: {combined.obs['sample_id'].nunique()}")
    print(f"protected genes kept: {present}/{len(coverage)}")


if __name__ == "__main__":
    main()
