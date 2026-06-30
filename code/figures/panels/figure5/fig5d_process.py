#!/usr/bin/env python3
"""Processing: 134 plasma biomarkers disease-significance ranking → fig5d_processed.csv

Ranks all 134 candidate plasma biomarkers by absolute disease correlation,
then flags the 4 features retained by the Network-Guided LASSO. Reveals that
LASSO captured the top-ranked biomarkers (BD-pTau-231 #4, pTau-231 #5, TREM1
#15) but also retained KLK6 (#119) — a feature with negligible univariate
correlation that contributes only through multivariate interaction.

Sources:
  output/step3_hub_identification/gene_significance/proteomics_gene_significance.csv
  output/step5_diagnostic_performance/panel_selected_features.csv
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

gs = pd.read_csv(PROJECT_ROOT / 'output' / 'step3_hub_identification' /
                 'gene_significance' / 'proteomics_gene_significance.csv')
sel = pd.read_csv(PROJECT_ROOT / 'output' / 'step5_diagnostic_performance' /
                  'panel_selected_features.csv')
ng_selected = set(sel[sel['panel'] == 'network_guided_l1']['feature'])

gs = gs.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
gs['rank'] = range(1, len(gs) + 1)
gs['selected'] = gs['gene'].isin(ng_selected)

rows = []
for _, r in gs.iterrows():
    rows.append({
        'gene': r['gene'],
        'correlation': float(r['correlation']),
        'pvalue': float(r['pvalue']),
        'abs_correlation': float(r['abs_correlation']),
        'rank': int(r['rank']),
        'selected': bool(r['selected']),
        'significant': bool(r['pvalue'] < 0.05),
    })

with open(os.path.join(OUT_DIR, 'fig5d_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['gene', 'correlation', 'pvalue',
                                      'abs_correlation', 'rank', 'selected',
                                      'significant'])
    w.writeheader()
    w.writerows(rows)

print(f'fig5d_process: ranked {len(rows)} biomarkers')
sel_rows = [r for r in rows if r['selected']]
for r in sel_rows:
    print(f"  LASSO-selected: {r['gene']:<14} rank={r['rank']:>3}/134  "
          f"|r|={r['abs_correlation']:.3f}  p={r['pvalue']:.2e}")
