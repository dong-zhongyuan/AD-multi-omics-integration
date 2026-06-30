#!/usr/bin/env python3
"""Processing: fig3b_raw.csv → fig3b_processed.csv
GenKI virtual-knockout perturbation footprint per hub gene.

For each knocked-out hub, record the number of downstream targets tested and
the number reaching statistical significance after knockout. This footprint
(coverage × specificity) is more interpretable than the raw KL divergence,
which is dominated by infinities (proteomics FABP3 / IGFBP7 / MAPT collapse to
degenerate distributions) and by a single transcriptomic outlier.

SNHG5 emerges as the perturbation with the broadest significant footprint
(14/44 targets), motivating its selection as the lead biological hit.

Source:
  fig3b_raw.csv (GenKI forward KO summaries, derived from
  output/step4_virtual_knockout/GenKI_NO3/*_statistics.csv)
"""
import os, csv
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(DIR, 'fig3b_raw.csv'))

# keep finite-KL + always the proteomics genes for completeness; flag the
# degenerate (inf/-inf) ones so the plot can grey them out
rows = []
for _, r in raw.iterrows():
    kl = r['KL_divergence_per_gene_mean']
    finite = pd.notna(kl) and kl not in (float('inf'), float('-inf'))
    rows.append({
        'KO_gene': r['KO_gene'],
        'tissue': 'transcriptomics' if r['KO_gene'] in ('SNHG5', 'PRKAR2B')
                  else 'proteomics',
        'n_target_genes': int(r['n_target_genes']),
        'n_significant_targets': int(r['n_significant_targets']),
        'kl_finite': bool(finite),
        'KL_divergence_per_gene_mean': (float(kl) if finite else None),
    })

# order by significant-target footprint (descending), then by total targets
rows.sort(key=lambda d: (d['n_significant_targets'], d['n_target_genes']),
          reverse=True)

with open(os.path.join(DIR, 'fig3b_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'KO_gene', 'tissue', 'n_target_genes', 'n_significant_targets',
        'kl_finite', 'KL_divergence_per_gene_mean'])
    w.writeheader()
    w.writerows(rows)

print('fig3b_process: KO perturbation footprint')
for r in rows:
    print(f"  {r['KO_gene']:<10} ({r['tissue']:<15}) "
          f"targets={r['n_target_genes']:>2}  significant={r['n_significant_targets']:>2}"
          f"  {'(finite KL)' if r['kl_finite'] else '(degenerate KL)'}")
