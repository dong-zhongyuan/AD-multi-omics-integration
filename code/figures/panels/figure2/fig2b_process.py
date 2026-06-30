#!/usr/bin/env python3
"""Processing: proteomics cross_tissue_edges.csv → fig2b_processed.csv

Top cross-tissue causal edges in the proteomics layer — the biologically
readable core of the brain↔blood causal network. Unlike the omics-wide hub
lists (anonymous Ensembl IDs / metabolite strings), proteomics nodes are named
proteins, so the strongest brain→blood / blood→brain edges can be interpreted
directly: tau-phosphorylation pairs (pTau-231↔pTau-181), MAPT hubs, neuropeptide
axes (VEGFD↔NPY, BDNF↔IL7), and amyloid fragments.

Selection: cross-tissue edges ranked by combined score
(strength × consistency), top 15 by max of the reciprocal pair to avoid
listing the same undirected edge twice.

Source:
  output/step2_cross_tissue_causality/proteomics/cross_tissue_edges.csv
"""
import os, csv
from pathlib import Path
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

edges = pd.read_csv(
    PROJECT_ROOT / 'output' / 'step2_cross_tissue_causality' / 'proteomics' /
    'cross_tissue_edges.csv')

# normalise mojibake / alias protein names so amyloid fragments read cleanly
NAME_FIX = {
    'A?38': 'Aβ38', 'AÎ²38': 'Aβ38', 'AÎ?38': 'Aβ38',
    'A?40': 'Aβ40', 'AÎ²40': 'Aβ40', 'AÎ?40': 'Aβ40',
    'A?42': 'Aβ42', 'AÎ²42': 'Aβ42', 'AÎ?42': 'Aβ42',
}
for col in ('source', 'target'):
    edges[col] = edges[col].replace(NAME_FIX)

ct = edges[edges['direction'] == 'cross_tissue'].copy()

# combined quality score for ranking
ct['score'] = ct['strength'] * ct['confidence_consistency']

# collapse reciprocal directed edges (A->B and B->A) into one undirected pair,
# keeping the stronger direction as representative; skip self-loops (alias pairs
# like Aβ40↔Aβ40 that are a single protein encoded under two aliases)
ct = ct[ct['source'] != ct['target']].copy()
ct['pair'] = ct.apply(
    lambda r: '—'.join(sorted([r['source'], r['target']])), axis=1)
best = (ct.sort_values('score', ascending=False)
          .drop_duplicates('pair', keep='first')
          .sort_values('score', ascending=False)
          .head(15)
          .reset_index(drop=True))
best['rank'] = range(1, len(best) + 1)

rows = []
for _, r in best.iterrows():
    rows.append({
        'rank': int(r['rank']),
        'source': r['source'],
        'target': r['target'],
        'edge': f"{r['source']} ↔ {r['target']}",
        'weight': float(r['weight']),
        'strength': float(r['strength']),
        'confidence_stability': float(r['confidence_stability']),
        'confidence_snr': float(r['confidence_snr']),
        'confidence_consistency': float(r['confidence_consistency']),
        'score': float(r['score']),
    })

with open(os.path.join(OUT_DIR, 'fig2b_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'rank', 'source', 'target', 'edge', 'weight', 'strength',
        'confidence_stability', 'confidence_snr', 'confidence_consistency',
        'score'])
    w.writeheader()
    w.writerows(rows)

print(f'fig2b_process: {len(rows)} top proteomics cross-tissue edges')
for r in rows[:8]:
    print(f"  #{r['rank']:>2}  {r['edge']:<24} strength={r['strength']:.2f}  "
          f"consistency={r['confidence_consistency']:.2f}")
