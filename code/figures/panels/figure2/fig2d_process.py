#!/usr/bin/env python3
"""Processing: edge count data → fig2d_processed.csv
Edge counts by directionality (brain-internal, blood-internal, cross-tissue) x 3 omics.
"""
import os, csv
from pathlib import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / 'output' / 'step2_cross_tissue_causality'

omics_list = ['transcriptomics', 'proteomics', 'metabolomics']
rows = []
for omics in omics_list:
    for direction, fname in [('Brain-internal', 'brain_network_edges.csv'),
                              ('Blood-internal', 'blood_network_edges.csv'),
                              ('Cross-tissue', 'cross_tissue_edges.csv')]:
        path = BASE / omics / 'filtered_edges' / fname
        n = 0
        with open(path) as f:
            n = sum(1 for _ in f) - 1  # minus header
        rows.append({'omics': omics, 'direction': direction, 'edge_count': n})

with open(os.path.join(OUT_DIR, 'fig2d_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'Wrote {len(rows)} rows')
