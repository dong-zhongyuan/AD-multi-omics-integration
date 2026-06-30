#!/usr/bin/env python3
"""Processing: step2 stats.json → fig2c_processed.csv

Network-scale fingerprint of each omics layer — features, samples, and edge
counts across the three network compartments (brain-internal, blood-internal,
cross-tissue). Complements fig2a (per-edge confidence *distributions*) with a
layer-level *scale* view: how the method scales from 132 plasma proteins to
2,000 transcriptomic features, and the resulting network sizes.

Source:
  output/step2_cross_tissue_causality/{omics}/stats.json
"""
import os, json, csv
from pathlib import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OMICS = [
    ('Transcriptomics', 'transcriptomics'),
    ('Proteomics', 'proteomics'),
    ('Metabolomics', 'metabolomics'),
]

rows = []
for label, key in OMICS:
    with open(PROJECT_ROOT / 'output' / 'step2_cross_tissue_causality' / key /
              'stats.json') as f:
        s = json.load(f)
    rows.append({
        'layer': label,
        'n_features': int(s['n_features']),
        'n_plasma_samples': int(s['n_samples']['plasma']),
        'n_csf_samples': int(s['n_samples']['csf']),
        'n_brain_edges': int(s['brain_network']['n_edges']),
        'n_blood_edges': int(s['blood_network']['n_edges']),
        'n_cross_tissue_edges': int(s['cross_tissue']['n_edges']),
        'brain_density': float(s['brain_network']['edge_density']),
        'blood_density': float(s['blood_network']['edge_density']),
        'cross_tissue_density': float(s['cross_tissue']['edge_density']),
    })

with open(os.path.join(OUT_DIR, 'fig2c_processed.csv'), 'w', newline='') as f:
    fields = ['layer', 'n_features', 'n_plasma_samples', 'n_csf_samples',
              'n_brain_edges', 'n_blood_edges', 'n_cross_tissue_edges',
              'brain_density', 'blood_density', 'cross_tissue_density']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print('fig2c_process: network-scale fingerprint for 3 omics layers')
for r in rows:
    print(f"  {r['layer']:<16} feat={r['n_features']:>5}  "
          f"brain/blood/cross edges = {r['n_brain_edges']}/"
          f"{r['n_blood_edges']}/{r['n_cross_tissue_edges']}")
