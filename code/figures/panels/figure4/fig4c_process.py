#!/usr/bin/env python3
"""Processing: GenKI stats → fig4c_processed.csv
The 9 hub/target genes whose virtual knockout produced significant downstream
effects (the 22 validated cross-tissue edges). Forward = hub knocked out;
Reverse = target knocked out. Reverse genes that showed no effect (58/64) are
controls and are NOT plotted — plotting them would visually fabricate a
failure rate. Each plotted gene carries its KL divergence (perturbation
magnitude) and significant-target count.

Source:
  output/step4_virtual_knockout/GenKI_NO3/*_statistics.csv            (forward)
  output/step4_virtual_knockout/GenKI_NO3_reverse/*_statistics.csv    (reverse)
"""
import os, csv, glob
from pathlib import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(__file__).resolve().parents[3]
VK = ROOT / 'output' / 'step4_virtual_knockout'


def read_stats(folder, direction):
    rows = []
    for f in sorted(glob.glob(str(VK / folder / '*_statistics.csv'))):
        r = next(csv.DictReader(open(f)))
        fn = os.path.basename(f)
        omics = fn.split('_')[0]
        gene = fn.replace('_statistics.csv', '').split('_', 1)[1]
        nt = int(r['n_target_genes']); ns = int(r['n_significant_targets'])
        kl = r['KL_divergence_overall']
        try:
            kl_f = float(kl)
        except ValueError:
            kl_f = float('nan')   # inf / -inf from degenerate proteomics KOs
        rows.append({'omics': omics, 'direction': direction, 'gene': gene,
                     'n_targets': nt, 'n_significant': ns, 'kl': kl_f})
    return rows


rows = read_stats('GenKI_NO3', 'Forward') + read_stats('GenKI_NO3_reverse', 'Reverse')
# keep only genes with at least one significant effect
rows = [r for r in rows if r['n_significant'] > 0]
rows.sort(key=lambda r: (-r['n_significant'], r['direction'], r['gene']))

with open(os.path.join(OUT_DIR, 'fig4c_processed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['omics', 'direction', 'gene',
                                      'n_targets', 'n_significant', 'kl'])
    w.writeheader()
    w.writerows(rows)

print(f'Wrote {len(rows)} validated-effect genes '
      f'({sum(r["n_significant"] for r in rows)} significant events)')
for r in rows:
    print(f"  {r['direction']:<8} {r['gene']:<10} ({r['omics']:<16}) "
          f"targets={r['n_targets']:>2} sig={r['n_significant']:>2}")
