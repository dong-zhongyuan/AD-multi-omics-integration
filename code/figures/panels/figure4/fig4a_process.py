#!/usr/bin/env python3
"""Processing: cross-tissue coexpression validation → fig4a_processed.csv

For figure-4 (knockout & causal validation): validates the predicted causal
edges in an independent AD-vs-Control cohort by comparing their co-expression
strength. Edges predicted by the pipeline should show stronger co-expression
in AD than in controls (disease-specific reinforcement).

Source: output/step6_external_validation/results/coexpression_validation.csv
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

df = pd.read_csv(PROJECT_ROOT / 'output' / 'step6_external_validation' /
                 'results' / 'coexpression_validation.csv')

rows = []
for _, r in df.iterrows():
    rows.append({
        'source': r['source'],
        'target': r['target'],
        'inph_weight': float(r['iPNH_weight']),
        'ad_r': float(r['AD_r']),
        'ctrl_r': float(r['CTRL_r']),
        'r_diff': float(r['r_diff']),
        'fisher_p': float(r['fisher_p']),
        'significant': bool(r['fisher_p'] < 0.05),
    })

# sort by r_diff descending (strongest disease-specific reinforcement first)
rows.sort(key=lambda x: x['r_diff'], reverse=True)

with open(os.path.join(OUT_DIR, 'fig4a_processed.csv'), 'w', newline='') as f:
    fieldnames = ['source', 'target', 'inph_weight', 'ad_r', 'ctrl_r',
                  'r_diff', 'fisher_p', 'significant']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f'fig4a_process: wrote {len(rows)} coexpression-validated edges')
sig = sum(r['significant'] for r in rows)
print(f'  significant (fisher_p<0.05): {sig}/{len(rows)}')
print(f'  r_diff range: {min(r["r_diff"] for r in rows):.3f} - '
      f'{max(r["r_diff"] for r in rows):.3f}')
