#!/usr/bin/env python3
"""Processing: roc_validation_results.csv → fig7d_processed.csv

NEW DESIGN: brain-specificity of the iNPH signature — brain (validated) vs
blood/PBMC (negative control).

The original fig7d was a "multi-category direction concordance" built on the
same flawed logFC-direction logic as fig7b/c (underlying r ≈ -0.06).

This replaces it with a positive-vs-negative control contrast: the Full
iNPH signature discriminates AD in brain tissue (AUC ~0.67-0.71, significant)
but NOT in blood/PBMC (AUC ~0.51, n.s.). This "brain-specific" pattern is
itself a meaningful result — the cross-tissue brain signal does not spill into
circulating immune cells.
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(OUT_DIR).resolve().parents[2]
SRC = PROJECT_ROOT / 'output' / 'step6_external_validation' / 'results' / 'roc_validation_results.csv'

df = pd.read_csv(SRC)

# collapse to tissue-level summary using the Full signature (most genes, broadest)
full = df[df['signature'] == 'Full_signature'].copy()

tissue_label = {
    'All_brain': 'All brain', 'BA9': 'BA9', 'Entorhinal_cortex': 'Entorhinal',
    'PBMC': 'PBMC (blood)',
}
# tissue group: brain vs blood
group = {'All_brain': 'Brain', 'BA9': 'Brain', 'Entorhinal_cortex': 'Brain',
         'PBMC': 'Blood'}
order = ['All_brain', 'BA9', 'Entorhinal_cortex', 'PBMC']

rows = []
for t in order:
    r = full[full['tissue'] == t].iloc[0]
    rows.append({
        'tissue': tissue_label[t],
        'group': group[t],
        'auc': round(float(r['auc']), 3),
        'ci_lower': round(float(r['ci_lower']), 3),
        'ci_upper': round(float(r['ci_upper']), 3),
        'perm_p': float(r['perm_pvalue']),
        'n_samples': int(r['n_samples']),
        'sig': bool(r['perm_pvalue'] < 0.05),
    })

with open(os.path.join(OUT_DIR, 'fig7d_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print('fig7d: brain vs blood AUC (Full signature)')
for r in rows:
    print(f"  {r['tissue']:<16} ({r['group']:<5}) AUC={r['auc']:.3f} "
          f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}] p={r['perm_p']:.4f} "
          f"{'*' if r['sig'] else 'ns'}")
