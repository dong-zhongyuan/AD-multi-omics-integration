#!/usr/bin/env python3
"""Processing: verified cross-tissue causal edges → fig3f_processed.csv

For figure-3 (hub identification & causal validation): scatter of causal
prediction rank vs KO validation effect size for the 22 verified edges.
Shows that the pipeline's causal predictions hold up under virtual knockout —
even edges ranked low (rank 40-63) produce significant KO effects.

Source: output/verified_cross_tissue_edges.csv
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

v = pd.read_csv(PROJECT_ROOT / 'output' / 'verified_cross_tissue_edges.csv')

rows = []
for _, r in v.iterrows():
    rows.append({
        'ko_gene': r['ko_gene'],
        'validated_gene': r['validated_gene'],
        'omics': r['omics'],
        'direction': 'Forward' if 'Forward' in r['direction'] else 'Reverse',
        'predicted_rank': int(r['predicted_rank']),
        'predicted_score': float(r['predicted_score']),
        'validation_effect_size': float(r['validation_effect_size']),
        'validation_p_value': float(r['validation_p_value']),
    })

with open(os.path.join(OUT_DIR, 'fig3f_processed.csv'), 'w', newline='') as f:
    fieldnames = ['ko_gene', 'validated_gene', 'omics', 'direction',
                  'predicted_rank', 'predicted_score',
                  'validation_effect_size', 'validation_p_value']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f'fig3f_process: wrote {len(rows)} verified causal edges')
print(f'  rank range: {min(r["predicted_rank"] for r in rows)}-'
      f'{max(r["predicted_rank"] for r in rows)}')
print(f'  effect size range: {min(r["validation_effect_size"] for r in rows):.2f}-'
      f'{max(r["validation_effect_size"] for r in rows):.2f}')
