#!/usr/bin/env python3
"""
根据动态验证结果分类 step5 候选基因

规则：
- 使用 output/verified_cross_tissue_edges.csv
- 只基于当前显著验证边
- 每条显著边的 source / target 两端都进入对应方向的 step5 候选集
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent


def expand_both_endpoints(verified_df: pd.DataFrame, direction: str, target_type: str, analysis: str) -> pd.DataFrame:
    sub = verified_df[verified_df["direction"] == direction].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "omics",
                "gene",
                "edge_partner",
                "endpoint_role",
                "validation",
                "type",
                "analysis",
                "predicted_score",
                "validation_p_value",
            ]
        )

    source_rows = sub[["omics", "source", "target", "predicted_score", "validation_p_value"]].copy()
    source_rows.columns = ["omics", "gene", "edge_partner", "predicted_score", "validation_p_value"]
    source_rows["endpoint_role"] = "source"

    target_rows = sub[["omics", "target", "source", "predicted_score", "validation_p_value"]].copy()
    target_rows.columns = ["omics", "gene", "edge_partner", "predicted_score", "validation_p_value"]
    target_rows["endpoint_role"] = "target"

    out = pd.concat([source_rows, target_rows], ignore_index=True)
    out["validation"] = direction
    out["type"] = target_type
    out["analysis"] = analysis
    out = out.drop_duplicates(subset=["omics", "gene", "endpoint_role", "validation"]).reset_index(drop=True)
    return out


def classify_genes():
    verified_file = PROJECT_ROOT / "output" / "verified_cross_tissue_edges.csv"
    verified_df = pd.read_csv(verified_file)

    diagnostic_df = expand_both_endpoints(
        verified_df,
        "Forward (CSF→PBMC)",
        "diagnostic",
        "diagnostic_performance",
    )
    therapeutic_df = expand_both_endpoints(
        verified_df,
        "Reverse (PBMC→CSF)",
        "therapeutic",
        "survival,gbd,nhanes,drugmining",
    )

    output_dir = PROJECT_ROOT / "output" / "step5_gene_classification"
    output_dir.mkdir(exist_ok=True)

    diagnostic_df.to_csv(output_dir / "diagnostic_targets.csv", index=False)
    therapeutic_df.to_csv(output_dir / "therapeutic_targets.csv", index=False)

    diagnostic_genes = diagnostic_df["gene"].dropna().astype(str).drop_duplicates().tolist()
    therapeutic_genes = therapeutic_df["gene"].dropna().astype(str).drop_duplicates().tolist()
    all_genes = sorted(set(diagnostic_genes) | set(therapeutic_genes))

    with open(output_dir / "diagnostic_genes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(diagnostic_genes))
    with open(output_dir / "therapeutic_genes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(therapeutic_genes))
    with open(output_dir / "all_step5_genes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_genes))

    pd.DataFrame({"gene": all_genes}).to_csv(output_dir / "all_step5_genes.csv", index=False)

    print("=" * 80)
    print("动态 step5 基因分类结果")
    print("=" * 80)
    print(f"\n【诊断候选】数量: {len(diagnostic_genes)}")
    print(", ".join(diagnostic_genes))
    print(f"\n【治疗候选】数量: {len(therapeutic_genes)}")
    print(", ".join(therapeutic_genes))
    print(f"\n【总候选】数量: {len(all_genes)}")
    print(f"\n已保存到: {output_dir}")

    return diagnostic_df, therapeutic_df


if __name__ == "__main__":
    classify_genes()
