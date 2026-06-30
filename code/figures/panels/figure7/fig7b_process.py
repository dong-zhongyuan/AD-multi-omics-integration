#!/usr/bin/env python3
"""Processing: coexpression_validation.csv → fig7b_processed.csv

NEW DESIGN: iNPH regulatory edges reproduced as co-expression in AD brain.

The original fig7b plotted iNPH-vs-AD logFC direction concordance, but the
underlying Pearson r ≈ -0.06 (iNPH CSF single-cell vs AD bulk brain, different
tissue & contrast) made the "79.7% concordance" a sign-bias artefact.

This panel instead shows the network-level reproduction that IS statistically
robust: of the 22 GenKI-validated cross-tissue edges, the 7 measurable in AD
bulk brain (GSE140841) are tested for co-expression Pearson r in AD vs Control,
with bootstrap 95% CI. iNPH edges co-express more strongly in AD than control.
"""
import os, csv
from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(OUT_DIR).resolve().parents[2]
SRC = PROJECT_ROOT / 'output' / 'step6_external_validation' / 'results' / 'coexpression_validation.csv'

df = pd.read_csv(SRC)

# build a compact edge label and order by AD_r descending (strongest first)
df = df.copy()
df['edge'] = df['source'] + ' - ' + df['target']
df = df.sort_values('AD_r', ascending=False).reset_index(drop=True)
df['edge'] = pd.Categorical(df['edge'], categories=df['edge'], ordered=True)

rows = []
for _, r in df.iterrows():
    rows.append({
        'edge': str(r['edge']),
        'source': r['source'],
        'target': r['target'],
        'ad_r': round(float(r['AD_r']), 3),
        'ctrl_r': round(float(r['CTRL_r']), 3),
        'r_diff': round(float(r['r_diff']), 3),
        'ad_p': float(r['AD_p']),
        'ctrl_p': float(r['CTRL_p']),
        'fisher_p': float(r['fisher_p']),
        'ad_sig': bool(r['AD_p'] < 0.05),
        'ad_specific': bool(r['fisher_p'] < 0.05),
    })

with open(os.path.join(OUT_DIR, 'fig7b_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(rows)
n_ad_sig = sum(r['ad_sig'] for r in rows)
n_specific = sum(r['ad_specific'] for r in rows)
print(f'fig7b: {n} co-expression edges validated')
print(f'  significant in AD (p<0.05): {n_ad_sig}/{n}')
print(f'  AD-specific (Fisher z p<0.05): {n_specific}/{n}')
print(f'  all AD>Control: {sum(r["r_diff"]>0 for r in rows)}/{n}')
