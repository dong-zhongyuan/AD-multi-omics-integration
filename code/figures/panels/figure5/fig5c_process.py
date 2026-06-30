#!/usr/bin/env python3
"""Processing: diagnostic-panel AUC → fig5c_processed.csv
Simple AUC bar chart source for the 4 LASSO diagnostic panels.
Source: output/step5_diagnostic_performance/panel_auc_summary.csv
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

summary = pd.read_csv(PROJECT_ROOT / 'output' / 'step5_diagnostic_performance' /
                      'panel_auc_summary.csv')

PANEL_LABEL = {
    'strict_forward':     'Strict-Forward',
    'verified_proteomics': 'Verified Proteomics',
    'network_guided_l1':  'Network-Guided',
    'full_plasma_l1':     'Full Plasma',
}

def feat_n(row):
    n = row.get('selected_n')
    if pd.notna(n) and n > 0:
        return int(n)
    return int(row['n_features'])

rows = []
for _, row in summary.iterrows():
    rows.append({
        'panel': row['panel'],
        'label': PANEL_LABEL.get(row['panel'], row['panel']),
        'n_features': feat_n(row),
        'cv_auc_mean': float(row['cv_auc_mean']),
        'cv_auc_sd': float(row['cv_auc_sd']),
    })

rows.sort(key=lambda r: r['cv_auc_mean'])

with open(os.path.join(OUT_DIR, 'fig5c_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['panel', 'label', 'n_features',
                                      'cv_auc_mean', 'cv_auc_sd'])
    w.writeheader()
    w.writerows(rows)

print(f'fig5c_process: wrote {len(rows)} panels')
