#!/usr/bin/env python3
"""
根据 VK 结果分类 step5 候选基因

规则：
- Forward VK（KO 脑端 → 观察血端）：
    血端受影响显著的基因（Z>2）→ 诊断标志物候选（血端可测）
- Reverse VK（KO 血端 → 观察脑端）：
    血端 KO 基因（n_sig>0）→ 治疗靶点候选（给药对象 → 药物挖掘）

蛋白组双 VK（GenKI + PPI）：
- 治疗靶点输出交集和并集两套，供下游选择

输出：
  - diagnostic_genes_{omics}.csv（血端诊断标志物候选）
  - therapeutic_KO_genes_{omics}.csv（血端治疗靶点候选 → 药物挖掘）
  - 蛋白组额外：therapeutic_KO_genes_proteomics_intersect.csv, _union.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

GENKI_FWD = PROJECT_ROOT / "output/step4_virtual_knockout/GenKI_NO3"
GENKI_REV = PROJECT_ROOT / "output/step4_virtual_knockout/GenKI_NO3_reverse"
PPI_FWD = PROJECT_ROOT / "output/step4_virtual_knockout/PPI_propagation/forward"
PPI_REV = PROJECT_ROOT / "output/step4_virtual_knockout/PPI_propagation/reverse"
OUTPUT_DIR = PROJECT_ROOT / "output/step5_gene_classification"


def load_genki_sig_ko(genki_dir: Path, omics: str) -> list:
    """从 GenKI statistics 提取 n_significant_targets > 0 的 KO 基因"""
    genes = []
    for f in sorted(genki_dir.glob(f"{omics}_*_statistics.csv")):
        gene = f.stem.replace(f"{omics}_", "").replace("_statistics", "")
        df = pd.read_csv(f)
        if len(df) > 0 and df.iloc[0]["n_significant_targets"] > 0:
            genes.append(gene)
    return genes


def load_genki_sig_targets(genki_dir: Path, omics: str) -> list:
    """从 GenKI gene_ranking 提取受影响显著的靶点基因（Z_score > 2）"""
    target_genes = set()
    for f in sorted(genki_dir.glob(f"{omics}_*_statistics.csv")):
        gene = f.stem.replace(f"{omics}_", "").replace("_statistics", "")
        df = pd.read_csv(f)
        if len(df) == 0 or df.iloc[0]["n_significant_targets"] == 0:
            continue
        ranking_file = genki_dir / f"{omics}_{gene}_gene_ranking.csv"
        if ranking_file.exists():
            rdf = pd.read_csv(ranking_file)
            if "Z_score" in rdf.columns:
                sig = rdf[rdf["Z_score"] > 2.0]["Gene"].tolist()
                target_genes.update(sig)
    return sorted(target_genes)


def load_ppi_sig_ko(ppi_dir: Path) -> list:
    """从 PPI propagation statistics 提取显著 KO 基因"""
    stats_files = list(ppi_dir.glob("*_statistics.csv"))
    if not stats_files:
        return []
    df = pd.read_csv(stats_files[0])
    return df[df["n_significant_targets"] > 0]["KO_gene"].tolist()


def classify():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_diag = []
    all_ther = []

    for omics in ["proteomics", "transcriptomics"]:
        # === 诊断标志物：GenKI Forward 受影响的血端靶点 ===
        diag_genes = load_genki_sig_targets(GENKI_FWD, omics)

        # === 治疗靶点：GenKI Reverse 的血端 KO 基因 ===
        genki_rev = load_genki_sig_ko(GENKI_REV, omics)

        # 蛋白组：PPI 传播是第二 VK，输出交集和并集
        if omics == "proteomics" and PPI_REV.exists():
            ppi_rev = load_ppi_sig_ko(PPI_REV)
            ther_intersect = sorted(set(genki_rev) & set(ppi_rev))
            ther_union = sorted(set(genki_rev) | set(ppi_rev))

            print(f"{'=' * 60}")
            print(f"{omics}")
            print(f"  诊断标志物（GenKI Forward 血端）: {len(diag_genes)} 基因")
            print(f"  治疗靶点 GenKI reverse: {len(genki_rev)} - {genki_rev}")
            print(f"  治疗靶点 PPI reverse: {len(ppi_rev)} - {ppi_rev}")
            print(f"  治疗靶点 交集: {len(ther_intersect)} - {ther_intersect}")
            print(f"  治疗靶点 并集: {len(ther_union)} - {ther_union}")

            # 保存交集和并集
            pd.DataFrame({"gene": ther_intersect}).to_csv(
                OUTPUT_DIR / f"therapeutic_KO_genes_{omics}_intersect.csv", index=False
            )
            pd.DataFrame({"gene": ther_union}).to_csv(
                OUTPUT_DIR / f"therapeutic_KO_genes_{omics}_union.csv", index=False
            )
            # 默认用交集
            ther_genes = ther_intersect
        else:
            ther_genes = genki_rev
            print(f"{'=' * 60}")
            print(f"{omics}")
            print(f"  诊断标志物（GenKI Forward 血端）: {len(diag_genes)} 基因")
            print(f"  治疗靶点（GenKI Reverse 血端KO）: {len(ther_genes)} 基因")
            print(f"    {ther_genes}")

        # 保存
        pd.DataFrame({"gene": diag_genes}).to_csv(
            OUTPUT_DIR / f"diagnostic_genes_{omics}.csv", index=False
        )
        pd.DataFrame({"gene": ther_genes}).to_csv(
            OUTPUT_DIR / f"therapeutic_KO_genes_{omics}.csv", index=False
        )

        all_diag.extend([{"omics": omics, "gene": g} for g in diag_genes])
        all_ther.extend([{"omics": omics, "gene": g} for g in ther_genes])

    # 合并
    diag_df = pd.DataFrame(all_diag).drop_duplicates(subset=["gene"])
    ther_df = pd.DataFrame(all_ther).drop_duplicates(subset=["gene"])
    diag_df.to_csv(OUTPUT_DIR / "diagnostic_genes.csv", index=False)
    ther_df.to_csv(OUTPUT_DIR / "therapeutic_KO_genes.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"总计：诊断标志物 {len(diag_df)} 基因，治疗靶点 {len(ther_df)} 基因")
    print(f"蛋白组交集/并集文件已单独保存")


if __name__ == "__main__":
    classify()
