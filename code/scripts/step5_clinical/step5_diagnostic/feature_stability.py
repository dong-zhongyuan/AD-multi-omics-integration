#!/usr/bin/env python3
"""Feature-selection stability for the network-guided diagnostic panel (reviewer M5).

Shows the 4-protein panel is STABLE: run the identical LASSO pipeline inside each fold
of RepeatedStratifiedKFold (5 splits x 20 repeats = 100 fits) and report how often each
candidate protein is retained. The CV is leakage-free: LASSO selection happens INSIDE
each training fold (Pipeline fit on train only), so test-fold data never informs selection.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.step5_clinical.step5_diagnostic.run_diagnostic_panels import (
    load_subject_level_plasma, get_step5_gene_sets,
)

OUT = Path(__file__).resolve().parents[3] / "output" / "step5_diagnostic_performance"
OUT.mkdir(parents=True, exist_ok=True)

df = load_subject_level_plasma()
gene_sets = get_step5_gene_sets()
features = [f for f in gene_sets["network_guided_pool"] if f in df.columns]
print(f"network-guided candidates: {len(features)} proteins")

X = df[features].fillna(df[features].median()).to_numpy()
y = df["diagnosis"].to_numpy()
print(f"subjects: {len(y)} (AD={int(y.sum())}, CN={int((y==0).sum())})")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
selection_counts = {f: 0 for f in features}
fold_aucs = []
n_fits = 0

for train_idx, test_idx in cv.split(X, y):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=8000, class_weight="balanced",
                                  solver="saga", penalty="l1", C=0.1)),
    ])
    model.fit(X_tr, y_tr)
    coef = model.named_steps["lr"].coef_[0]
    for j, f in enumerate(features):
        if coef[j] != 0:
            selection_counts[f] += 1
    proba = model.predict_proba(X_te)[:, 1]
    fold_aucs.append(roc_auc_score(y_te, proba))
    n_fits += 1

print(f"\ntotal fits: {n_fits}")
stab = (pd.DataFrame({"feature": list(selection_counts.keys()),
                      "selection_freq": [selection_counts[f]/n_fits for f in features]})
        .sort_values("selection_freq", ascending=False))
stab = stab[stab["selection_freq"] > 0].reset_index(drop=True)
stab.to_csv(OUT / "feature_stability.csv", index=False)
print("\n=== Feature selection stability (top 10) ===")
print(stab.head(10).to_string(index=False))
print(f"\nfold AUC: mean={np.mean(fold_aucs):.3f} sd={np.std(fold_aucs):.3f} "
      f"[{np.percentile(fold_aucs,2.5):.3f}, {np.percentile(fold_aucs,97.5):.3f}]")
