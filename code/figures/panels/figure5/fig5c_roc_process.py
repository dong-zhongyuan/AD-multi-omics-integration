#!/usr/bin/env python3
"""Processing: extract ROC curve points for diagnostic panels → fig5c_processed.csv

Reproduces the 4 LASSO diagnostic panels from run_diagnostic_panels.py but uses
cross_val_predict to obtain out-of-fold predicted probabilities, then computes
ROC curve points (fpr, tpr) for each panel.

Source: scripts/step5_clinical/step5_diagnostic/run_diagnostic_panels.py (model logic)
         data/blood-transcription-protein/ (plasma data)
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "step5_clinical" / "step5_diagnostic"))
import run_diagnostic_panels as rdp
# fix hardcoded WSL path for Windows
rdp.PROJECT_ROOT = PROJECT_ROOT
from run_diagnostic_panels import load_subject_level_plasma, get_step5_gene_sets

df = load_subject_level_plasma()
gene_sets = get_step5_gene_sets()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_predict = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # for OOF predictions
C_VAL = 0.1

PANEL_LABEL = {
    "strict_forward": "Strict-Forward",
    "verified_proteomics": "Verified Proteomics",
    "network_guided_l1": "Network-Guided",
    "full_plasma_l1": "Full Plasma",
}

def make_model(mode, c_val):
    if mode == "fixed":
        return Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=8000, class_weight="balanced",
                                               solver="saga", penalty="l1", C=c_val))])

def panel_features(panel_key):
    if panel_key == "strict_forward":
        return gene_sets["strict_forward"]
    if panel_key == "verified_proteomics":
        return gene_sets["verified_proteomics"]
    if panel_key == "network_guided_l1":
        return gene_sets["network_guided_pool"]
    # full_plasma_l1
    return [c for c in df.columns if c not in {"RID", "DIAGNOSIS", "diagnosis"}]

rows = []
for panel_key, mode in [("strict_forward", "fixed"), ("verified_proteomics", "fixed"),
                        ("network_guided_l1", "l1"), ("full_plasma_l1", "l1")]:
    features = [f for f in panel_features(panel_key) if f in df.columns]
    if not features:
        continue
    panel_df = df[["diagnosis"] + features].copy()
    X = panel_df[features].fillna(panel_df[features].median()).to_numpy()
    y = panel_df["diagnosis"].to_numpy()
    model = make_model(mode, C_VAL)
    y_proba = cross_val_predict(model, X, y, cv=cv_predict, method="predict_proba", n_jobs=1)[:, 1]
    auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)
    label = PANEL_LABEL.get(panel_key, panel_key)
    n_feat = len(features)
    step = max(1, len(fpr) // 100)
    for i in range(0, len(fpr), step):
        rows.append({"panel": panel_key, "label": label, "auc": round(auc, 4),
                     "n_features": n_feat, "fpr": round(fpr[i], 5), "tpr": round(tpr[i], 5)})
    rows.append({"panel": panel_key, "label": label, "auc": round(auc, 4),
                 "n_features": n_feat, "fpr": 1.0, "tpr": 1.0})

with open(os.path.join(OUT_DIR, "fig5c_processed.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["panel", "label", "auc", "n_features", "fpr", "tpr"])
    w.writeheader()
    w.writerows(rows)

print("fig5c_roc_process: extracted ROC curves")
seen = {}
for r in rows:
    if r["panel"] not in seen:
        seen[r["panel"]] = r
for pk in ["strict_forward", "verified_proteomics", "network_guided_l1", "full_plasma_l1"]:
    if pk in seen:
        r = seen[pk]
        print(f"  {r['label']:<22} AUC={r['auc']:.3f}  ({r['n_features']} features)")
