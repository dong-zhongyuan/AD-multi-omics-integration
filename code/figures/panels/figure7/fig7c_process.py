#!/usr/bin/env python3
"""Processing: roc_validation_results.csv → fig7c_processed.csv

NEW DESIGN: discriminative power of the iNPH brain signature across brain
regions (BA9, Entorhinal cortex, All-brain).

The original fig7c was a "signal-dose consistency curve" built on the same
flawed logFC-direction logic as fig7b (underlying r ≈ -0.06). This replaces it
with the robust ROC-AUC evidence: for each brain region, the iNPH Brain /
Blood / Full signatures' AUC with bootstrap CI and permutation p-value.

Focus on brain regions (the positive, validated tissue) — this complements
fig7a's tissue×signature heatmap by isolating region-level stability.
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(OUT_DIR).resolve().parents[2]
SRC = PROJECT_ROOT / 'output' / 'step6_external_validation' / 'results' / 'roc_validation_results.csv'

df = pd.read_csv(SRC)

# brain regions only (BA9, Entorhinal_cortex, All_brain) — the validated tissue
brain = df[df['tissue'].isin(['BA9', 'Entorhinal_cortex', 'All_brain'])].copy()
brain['signature'] = (brain['signature']
                      .str.replace('_signature', '', regex=False)
                      .str.replace('_control', '', regex=False))

# ordered tissue factor: All_brain first (broadest), then specific regions
tissue_order = ['All_brain', 'BA9', 'Entorhinal_cortex']
tissue_label = {'All_brain': 'All brain', 'BA9': 'BA9', 'Entorhinal_cortex': 'Entorhinal'}
sig_order = ['Brain', 'Blood', 'Full']

rows = []
for _, r in brain.iterrows():
    rows.append({
        'tissue': tissue_label[r['tissue']],
        'signature': r['signature'],
        'n_genes': int(r['n_genes_used']),
        'n_samples': int(r['n_samples']),
        'auc': round(float(r['auc']), 3),
        'ci_lower': round(float(r['ci_lower']), 3),
        'ci_upper': round(float(r['ci_upper']), 3),
        'perm_p': float(r['perm_pvalue']),
    })

with open(os.path.join(OUT_DIR, 'fig7c_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f'fig7c: {len(rows)} brain-region AUCs (3 tissues x 3 signatures)')
for r in rows:
    print(f"  {r['tissue']:<16} {r['signature']:<6} AUC={r['auc']:.3f} "
          f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}] p={r['perm_p']:.4f}")
