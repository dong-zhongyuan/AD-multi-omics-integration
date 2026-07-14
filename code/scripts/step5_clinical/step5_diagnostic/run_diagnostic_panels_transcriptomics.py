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


def load_subject_level_expression(gene_list=None) -> pd.DataFrame:
    """高效加载 ADNI 基因表达微阵列。
    gene_list: 只加载这些基因（None=加载全部，不推荐）
    """
    expr_path = PROJECT_ROOT / "data/survival/ADNI_Gene_Expression_Profile.csv"

    # 读取 header 获取 SubjectID
    header_df = pd.read_csv(expr_path, nrows=3, low_memory=False)
    sid_row = header_df[header_df.iloc[:, 0] == 'SubjectID']
    subject_ids = sid_row.iloc[0, 3:].tolist()

    # 逐块过滤，只保留目标基因
    gene_set = set(gene_list) if gene_list else None
    kept_chunks = []
    for chunk in pd.read_csv(expr_path, skiprows=[1], chunksize=5000, low_memory=False):
        probe_mask = chunk.iloc[:, 0].astype(str).str.endswith('_at')
        if not probe_mask.any():
            continue
        probes = chunk[probe_mask].copy()
        if gene_set is not None:
            symbol_col = probes.columns[2]
            probes = probes[probes[symbol_col].isin(gene_set)]
        if not probes.empty:
            kept_chunks.append(probes)

    raw = pd.concat(kept_chunks, ignore_index=True)
    raw.columns = ['ProbeSet', 'LocusLink', 'Symbol'] + list(raw.columns[3:])
    expr = raw[['Symbol'] + list(raw.columns[3:])].copy()
    expr = expr[expr['Symbol'].notna() & (expr['Symbol'] != '')]
    for col in list(raw.columns[3:]):
        expr[col] = pd.to_numeric(expr[col], errors='coerce')
    expr = expr.groupby('Symbol').mean()

    # SubjectID → RID
    registry = pd.read_csv(PROJECT_ROOT / "data/survival/REGISTRY_05May2026.csv")
    sid_to_rid = registry[['PTID', 'RID']].drop_duplicates(subset='PTID').set_index('PTID')['RID'].to_dict()
    rid_list = [sid_to_rid.get(sid) for sid in subject_ids]
    expr.columns = rid_list
    expr = expr.loc[:, pd.notna(expr.columns)]
    expr.columns = [int(c) for c in expr.columns]

    # 转置为 RID × gene
    expr_t = expr.T.reset_index()
    expr_t.columns = ['RID'] + list(expr_t.columns[1:])
    expr_t['RID'] = expr_t['RID'].astype(int)

    # 诊断标签
    dx_df = pd.read_csv(PROJECT_ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")
    dx_df = dx_df.loc[dx_df["VISCODE2"] == "bl", ["RID", "DIAGNOSIS"]].copy()
    dx_df = dx_df[dx_df["DIAGNOSIS"].isin([1, 3])].copy()
    dx_df["diagnosis"] = (dx_df["DIAGNOSIS"] == 3).astype(int)

    merged = dx_df.merge(expr_t, on='RID', how='inner')
    return merged


def get_diag_genes() -> list[str]:
    """从 classify_validated_genes.py 生成的文件读取诊断标志物基因"""
    diag_file = PROJECT_ROOT / "output/step5_gene_classification/diagnostic_genes_transcriptomics.csv"
    if diag_file.exists():
        df = pd.read_csv(diag_file)
        return sorted(set(df["gene"].astype(str)))
    return []


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
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
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

    # 诊断标志物 = 从 classify_validated_genes.py 生成的文件读取
    diag_genes = get_diag_genes()
    if not diag_genes:
        print("❌ 没有找到 VK forward 诊断标志物基因")
        return

    print(f"加载 ADNI 微阵列（只读 {len(diag_genes)} 个诊断标志物基因）...")
    df = load_subject_level_expression(diag_genes)
    print(f"  合并后: {len(df)} 样本, {len(df.columns)-3} 基因")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    summary_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []

    # VK forward 血端标志物 Fixed panel
    result = evaluate_fixed_panel(df, "vk_forward_fixed", diag_genes, cv)
    if result is not None:
        summary_rows.append(result)

    # VK forward 血端标志物 + L1 稀疏化
    l1_result, l1_selected = evaluate_l1_panel(
        df, "vk_forward_l1", diag_genes, cv, c_value=0.1,
    )
    if l1_result is not None:
        summary_rows.append(l1_result)
    if l1_selected is not None:
        selected_rows.append(l1_selected)

    summary_df = pd.DataFrame(summary_rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    summary_file = OUTPUT_DIR / "panel_auc_summary_transcriptomics.csv"
    summary_df.to_csv(summary_file, index=False)

    if selected_rows:
        selected_df = pd.concat(selected_rows, ignore_index=True)
        selected_df.to_csv(OUTPUT_DIR / "panel_selected_features_transcriptomics.csv", index=False)

    meta = {
        "n_subjects": int(len(df)),
        "n_cn": int((df["diagnosis"] == 0).sum()),
        "n_ad": int((df["diagnosis"] == 1).sum()),
        "vk_forward_diag_genes": diag_genes,
    }
    with open(OUTPUT_DIR / "panel_run_metadata_transcriptomics.json", "w", encoding="utf-8") as f:
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

    all_panel_genes = [g for g in diag_genes if g in df.columns]

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
    single_df.to_csv(OUTPUT_DIR / "single_gene_auc_transcriptomics.csv", index=False)

    print(single_df.to_string(index=False))
    print(f"\n保存: {OUTPUT_DIR / 'single_gene_auc_transcriptomics.csv'}")


if __name__ == "__main__":
    main()
