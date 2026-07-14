#!/usr/bin/env python3
"""
不重跑 step4 的血浆诊断 panel 评估

策略：
- 统一使用 baseline CN vs AD 受试者级 PLASMA 蛋白矩阵
- 评估三类 panel：
  1. strict_forward: 当前 step5 诊断蛋白
  2. verified_proteomics: 所有已验证蛋白边两端基因
  3. network_guided_l1: 从 step3 蛋白跨组织网络里提取与验证蛋白相连的候选池，使用 L1 logistic 自动稀疏化
- 可选 full_plasma_l1 作为性能上限参考，但不建议作为主结果
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import os
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
OUTPUT_DIR = PROJECT_ROOT / "output" / "step5_diagnostic_performance"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_subject_level_plasma() -> pd.DataFrame:
    plasma_df = pd.read_csv(
        PROJECT_ROOT / "data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv",
        low_memory=False,
    )
    dx_df = pd.read_csv(PROJECT_ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")

    dx_df = dx_df.loc[dx_df["VISCODE"] == "bl", ["RID", "DIAGNOSIS"]].copy()
    dx_df = dx_df[dx_df["DIAGNOSIS"].isin([1, 3])].copy()
    dx_df["diagnosis"] = (dx_df["DIAGNOSIS"] == 3).astype(int)

    plasma_df = plasma_df[
        (plasma_df["SampleMatrixType"] == "PLASMA")
        & (plasma_df["SampleQC"] == "passed")
        & (plasma_df["VISCODE"] == "bl")
        & plasma_df["RID"].notna()
    ][["RID", "Target", "NPQ"]].copy()

    plasma_df["RID"] = pd.to_numeric(plasma_df["RID"], errors="coerce")
    plasma_df["NPQ"] = pd.to_numeric(plasma_df["NPQ"], errors="coerce")
    plasma_df = plasma_df.dropna(subset=["RID", "NPQ"])

    wide = plasma_df.pivot_table(
        index="RID",
        columns="Target",
        values="NPQ",
        aggfunc="mean",
    )

    merged = dx_df.merge(wide, left_on="RID", right_index=True, how="inner")
    return merged


def get_step5_gene_sets() -> dict[str, list[str]]:
    # 直接从 Step3 提取基因（不依赖 VK verified_cross_tissue_edges.csv）
    # 蛋白组 Hub 基因 + filtered edges 的所有基因
    edges_file = PROJECT_ROOT / "output" / "step3_hub_identification" / "eigengene_analysis" / "proteomics" / "filtered_cross_tissue_edges.csv"
    edges = pd.read_csv(edges_file)

    # 清洗蛋白名 → gene symbol
    def clean_name(name):
        name = str(name).upper()
        if name.startswith("BD-"):
            name = name[3:]
        if name.startswith("PTAU") or "PTAU" in name:
            return "MAPT"
        if name.startswith("ABETA") or name.startswith("Aβ"):
            return "APP"
        return name

    brain_genes = sorted(set(edges["source"].apply(clean_name)))
    blood_genes = sorted(set(edges["target"].apply(clean_name)))
    all_proteomics = sorted(set(brain_genes) | set(blood_genes))

    # 同时加载 hub_cross_tissue_edges 构建更大的 network-guided pool
    hub_edges_file = PROJECT_ROOT / "output" / "step3_hub_identification" / "proteomics" / "hub_cross_tissue_edges.csv"
    if hub_edges_file.exists():
        hub_edges = pd.read_csv(hub_edges_file)
        hub_genes = sorted(
            set(hub_edges["source"].apply(clean_name)) |
            set(hub_edges["target"].apply(clean_name))
        )
        network_guided = sorted(set(all_proteomics) | set(hub_genes))
    else:
        network_guided = all_proteomics

    # verified_proteomics = 脑端 Hub 基因（做 fixed panel）
    verified_proteomics = brain_genes

    # strict_forward = 血端疾病相关基因
    strict_forward = blood_genes

    return {
        "strict_forward": strict_forward,
        "verified_proteomics": verified_proteomics,
        "network_guided_pool": network_guided,
    }


def evaluate_fixed_panel(df: pd.DataFrame, panel_name: str, features: list[str], cv) -> dict | None:
    features = [f for f in features if f in df.columns]
    if not features:
        return None

    panel_df = df[["diagnosis"] + features].dropna().copy()
    X = panel_df[features].to_numpy()
    y = panel_df["diagnosis"].to_numpy()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    apparent_auc = None
    if len(features) == 1:
        apparent_auc = roc_auc_score(y, X[:, 0])
        apparent_auc = max(apparent_auc, 1 - apparent_auc)

    return {
        "panel": panel_name,
        "mode": "fixed",
        "n_subjects": int(len(panel_df)),
        "n_features": int(len(features)),
        "features": "|".join(features),
        "cv_auc_mean": float(scores.mean()),
        "cv_auc_sd": float(scores.std()),
        "apparent_auc": None if apparent_auc is None else float(apparent_auc),
    }


def evaluate_l1_panel(
    df: pd.DataFrame,
    panel_name: str,
    candidate_features: list[str],
    cv,
    c_value: float,
) -> tuple[dict | None, pd.DataFrame | None]:
    features = [f for f in candidate_features if f in df.columns]
    if not features:
        return None, None

    panel_df = df[["diagnosis"] + features].copy()
    X = panel_df[features].fillna(panel_df[features].median()).to_numpy()
    y = panel_df["diagnosis"].to_numpy()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=8000,
                    class_weight="balanced",
                    solver="saga",
                    penalty="l1",
                    C=c_value,
                ),
            ),
        ]
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    model.fit(X, y)
    coef = model.named_steps["lr"].coef_[0]
    selected = pd.DataFrame({"feature": features, "coef": coef})
    selected = selected[selected["coef"].abs() > 1e-8].copy()
    selected["panel"] = panel_name
    selected = selected.sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)

    result = {
        "panel": panel_name,
        "mode": "l1_pool",
        "n_subjects": int(len(panel_df)),
        "n_features": int(len(features)),
        "features": "|".join(features),
        "cv_auc_mean": float(scores.mean()),
        "cv_auc_sd": float(scores.std()),
        "apparent_auc": None,
        "selected_features": "|".join(selected["feature"].tolist()),
        "selected_n": int(len(selected)),
        "C": float(c_value),
    }
    return result, selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_subject_level_plasma()
    gene_sets = get_step5_gene_sets()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

    summary_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []

    for panel_name in ["strict_forward", "verified_proteomics"]:
        result = evaluate_fixed_panel(df, panel_name, gene_sets[panel_name], cv)
        if result is not None:
            summary_rows.append(result)

    network_result, network_selected = evaluate_l1_panel(
        df,
        "network_guided_l1",
        gene_sets["network_guided_pool"],
        cv,
        c_value=0.1,
    )
    if network_result is not None:
        summary_rows.append(network_result)
    if network_selected is not None:
        selected_rows.append(network_selected)

    full_result, full_selected = evaluate_l1_panel(
        df,
        "full_plasma_l1",
        [c for c in df.columns if c not in {"RID", "DIAGNOSIS", "diagnosis"}],
        cv,
        c_value=0.1,
    )
    if full_result is not None:
        summary_rows.append(full_result)
    if full_selected is not None:
        selected_rows.append(full_selected)

    summary_df = pd.DataFrame(summary_rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    summary_file = OUTPUT_DIR / "panel_auc_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    if selected_rows:
        selected_df = pd.concat(selected_rows, ignore_index=True)
        selected_df.to_csv(OUTPUT_DIR / "panel_selected_features.csv", index=False)

    meta = {
        "n_subjects": int(len(df)),
        "n_cn": int((df["diagnosis"] == 0).sum()),
        "n_ad": int((df["diagnosis"] == 1).sum()),
        "network_guided_pool_size": int(len(gene_sets["network_guided_pool"])),
        "strict_forward_genes": gene_sets["strict_forward"],
        "verified_proteomics_genes": gene_sets["verified_proteomics"],
    }
    with open(OUTPUT_DIR / "panel_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("诊断 panel 评估完成")
    print("=" * 80)
    print(summary_df[["panel", "mode", "cv_auc_mean", "cv_auc_sd", "n_subjects"]].to_string(index=False))
    print(f"\n结果已保存到: {summary_file}")

    # ================================================================
    # 单个基因诊断效力（单变量 AUC）
    # ================================================================
    print("\n" + "=" * 80)
    print("单个基因诊断效力")
    print("=" * 80)

    from sklearn.metrics import roc_auc_score

    all_panel_genes = sorted(set(
        gene_sets["network_guided_pool"] +
        gene_sets["verified_proteomics"] +
        gene_sets["strict_forward"]
    ))
    # 去掉非蛋白名
    all_panel_genes = [g for g in all_panel_genes if g in df.columns]

    single_results = []
    for gene in all_panel_genes:
        vals = df[["diagnosis", gene]].dropna()
        if len(vals) < 50:
            continue
        y = vals["diagnosis"].values
        x = vals[gene].values
        # AUC（取 max(AUC, 1-AUC) 因为方向不确定）
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
        except Exception:
            continue
        # 方向：正相关=风险（高值→AD），负相关=保护
        direction = "risk" if roc_auc_score(y, x) > 0.5 else "protective"
        single_results.append({
            "gene": gene,
            "single_auc": round(auc, 4),
            "direction": direction,
            "n": len(vals),
        })

    single_df = pd.DataFrame(single_results).sort_values("single_auc", ascending=False)
    single_df.to_csv(OUTPUT_DIR / "single_gene_auc.csv", index=False)

    print(single_df.to_string(index=False))
    print(f"\n保存: {OUTPUT_DIR / 'single_gene_auc.csv'}")


if __name__ == "__main__":
    main()
