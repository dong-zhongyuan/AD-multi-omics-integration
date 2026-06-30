#!/usr/bin/env python3
"""Processing: roc_validation_results.csv → fig7a_processed.csv
External validation AUC — brain tissue signature discrimination.
"""
import os, sys, csv
from pathlib import Path
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / 'output' / 'step6_external_validation' / 'results' / 'roc_validation_results.csv'

rows = []
with open(SRC) as f:
    for r in csv.DictReader(f):
        if r['tissue'] in ('BA9', 'Entorhinal_cortex', 'All_brain'):
            rows.append({
                'tissue': r['tissue'],
                'signature': r['signature'].replace('_signature', '').replace('_control', ''),
                'n_genes': int(r['n_genes_used']),
                'n_samples': int(r['n_samples']),
                'auc': float(r['auc']),
                'ci_lower': float(r['ci_lower']),
                'ci_upper': float(r['ci_upper']),
                'perm_p': float(r['perm_pvalue']),
            })

with open(os.path.join(OUT_DIR, 'fig7a_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'Wrote {len(rows)} rows to fig7a_processed.csv')
