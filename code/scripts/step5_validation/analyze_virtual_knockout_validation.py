#!/usr/bin/env python3
"""
动态汇总 GenKI 显著验证边

规则：
- 只读取当前 output/step4_virtual_knockout/GenKI_NO3 与 GenKI_NO3_reverse 的结果
- 只保留 negative control 分析里显著的边
- transcriptomics 先把 step3 的 Ensembl 边映射成 gene symbol，再和 GenKI 输出对齐

输出：
- output/verified_cross_tissue_edges.csv
- output/virtual_knockout_validation_report.md
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from tools.gene_id_converter import ensembl_to_symbol


def load_edges(omics: str) -> pd.DataFrame:
    """Load step3 edges and normalize transcriptomics IDs to symbols."""
    edge_file = PROJECT_ROOT / "output" / "step3_hub_identification" / "eigengene_analysis" / omics / "filtered_cross_tissue_edges.csv"
    df = pd.read_csv(edge_file).copy()
    df["source_raw"] = df["source"].astype(str)
    df["target_raw"] = df["target"].astype(str)

    if omics == "transcriptomics":
        ids = list(set(df["source_raw"]).union(set(df["target_raw"])))
        mapping = ensembl_to_symbol(ids)
        df["source"] = df["source_raw"].map(lambda x: mapping.get(x, x))
        df["target"] = df["target_raw"].map(lambda x: mapping.get(x, x))
    else:
        df["source"] = df["source_raw"]
        df["target"] = df["target_raw"]

    return df


def parse_nc_filename(path: Path):
    match = re.match(r"^(proteomics|transcriptomics)_(.+)_negative_controls\.csv$", path.name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def collect_genki_verified_edges(run_dir: Path, direction_label: str) -> list[dict]:
    """Collect significant verified edges from one GenKI direction directory."""
    verified = []

    for nc_file in sorted(run_dir.glob("*_negative_controls.csv")):
        omics, ko_gene = parse_nc_filename(nc_file)
        if omics is None:
            continue

        nc_df = pd.read_csv(nc_file)
        if "significant" not in nc_df.columns:
            continue
        sig_df = nc_df[nc_df["significant"] == True].copy()
        if sig_df.empty:
            continue

        edges_df = load_edges(omics)

        for _, row in sig_df.iterrows():
            target_gene = str(row["target_gene"])

            if direction_label == "Forward (CSF→PBMC)":
                matched = edges_df[
                    (edges_df["source"] == ko_gene) &
                    (edges_df["target"] == target_gene)
                ].copy()
            else:
                matched = edges_df[
                    (edges_df["source"] == target_gene) &
                    (edges_df["target"] == ko_gene)
                ].copy()

            if matched.empty:
                continue

            matched = matched.sort_values("final_score", ascending=False)
            best = matched.iloc[0]
            verified.append(
                {
                    "omics": omics,
                    "method": "GenKI",
                    "direction": direction_label,
                    "source": best["source"],
                    "target": best["target"],
                    "source_raw": best["source_raw"],
                    "target_raw": best["target_raw"],
                    "predicted_weight": best["weight"],
                    "predicted_score": best["final_score"],
                    "predicted_rank": int(best["rank"]),
                    "ko_gene": ko_gene,
                    "validated_gene": target_gene,
                    "validation_p_value": float(row["p_value"]),
                    "validation_effect_size": float(row["cohens_d"]),
                    "validation_significant": True,
                }
            )

    return verified


def generate_report(verified_df: pd.DataFrame):
    report_file = PROJECT_ROOT / "output" / "virtual_knockout_validation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 动态虚拟敲除验证报告\n\n")
        f.write(f"总计显著验证边: {len(verified_df)}\n\n")

        f.write("## 分组统计\n\n")
        for (omics, direction), sub in verified_df.groupby(["omics", "direction"]):
            f.write(f"- {omics} / {direction}: {len(sub)} 条\n")

        f.write("\n## 详细列表\n\n")
        f.write("| omics | direction | source | target | predicted_rank | predicted_score | p_value | effect_size |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|\n")
        for _, row in verified_df.sort_values(["omics", "direction", "predicted_rank"]).iterrows():
            f.write(
                f"| {row['omics']} | {row['direction']} | {row['source']} | {row['target']} | "
                f"{int(row['predicted_rank'])} | {row['predicted_score']:.6f} | "
                f"{row['validation_p_value']:.6f} | {row['validation_effect_size']:.6f} |\n"
            )


def main():
    print("=" * 80)
    print("动态虚拟敲除验证分析（GenKI only）")
    print("=" * 80)

    forward_dir = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3"
    reverse_dir = PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3_reverse"

    all_verified = []
    all_verified.extend(collect_genki_verified_edges(forward_dir, "Forward (CSF→PBMC)"))
    all_verified.extend(collect_genki_verified_edges(reverse_dir, "Reverse (PBMC→CSF)"))

    if not all_verified:
        print("未找到显著验证边")
        return

    verified_df = pd.DataFrame(all_verified)
    verified_df = verified_df.sort_values(
        ["omics", "direction", "predicted_rank", "validation_p_value"],
        ascending=[True, True, True, True],
    ).drop_duplicates(
        subset=["omics", "direction", "source", "target"], keep="first"
    ).reset_index(drop=True)

    output_file = PROJECT_ROOT / "output" / "verified_cross_tissue_edges.csv"
    verified_df.to_csv(output_file, index=False)

    print(f"\n总计显著验证边: {len(verified_df)}")
    print("\n按组学 / 方向:")
    print(verified_df.groupby(["omics", "direction"]).size())
    print(f"\n结果已保存到: {output_file}")

    generate_report(verified_df)
    print(f"报告已保存到: {PROJECT_ROOT / 'output' / 'virtual_knockout_validation_report.md'}")


if __name__ == "__main__":
    main()
