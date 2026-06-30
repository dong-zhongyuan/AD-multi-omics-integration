#!/usr/bin/env python3
"""Processing: SNHG5 KO significant targets → fig4b_processed.csv

Lollipop-chart source — Cohen's d for each of the 14 significant downstream
targets of the SNHG5 (transcriptomics, forward) virtual knockout, with the
negative-control reference band summarised across all 42 SNHG5 targets.

Source: output/step4_virtual_knockout/GenKI_NO3/transcriptomics_SNHG5_negative_controls.csv
"""
import os, csv
from pathlib import Path
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VK_BASE = PROJECT_ROOT / 'output' / 'step4_virtual_knockout'

nc_path = VK_BASE / 'GenKI_NO3' / 'transcriptomics_SNHG5_negative_controls.csv'

rows = []
with open(nc_path) as fh:
    r = csv.DictReader(fh)
    for row in r:
        try:
            is_sig = str(row.get('significant', '')).strip() == 'True'
            rows.append({
                'target_gene': row['target_gene'],
                'target_kl': float(row['target_kl']),
                'n_controls': int(row.get('n_controls', 0)),
                'control_kl_mean': float(row.get('control_kl_mean', 0)),
                'control_kl_std': float(row.get('control_kl_std', 0)),
                'cohens_d': float(row['cohens_d']),
                'p_value': float(row['p_value']),
                'significant': is_sig,
            })
        except (ValueError, KeyError):
            pass

# Negative-control reference band: mean & SD of Cohen's d across ALL targets
# (the null distribution of effect sizes expected from expression-matched controls).
all_d = np.array([r['cohens_d'] for r in rows], dtype=float)
ctrl_band_mean = float(np.mean(all_d))
ctrl_band_std = float(np.std(all_d))
ctrl_band_p95 = float(np.percentile(all_d, 95))

# Keep only significant targets, sort largest → smallest for the lollipop chart
sig_rows = [r for r in rows if r['significant']]
sig_rows.sort(key=lambda r: r['cohens_d'], reverse=True)

with open(os.path.join(OUT_DIR, 'fig4b_processed.csv'), 'w', newline='') as f:
    fieldnames = ['target_gene', 'target_kl', 'n_controls',
                  'control_kl_mean', 'control_kl_std',
                  'cohens_d', 'p_value', 'significant',
                  'ctrl_band_mean', 'ctrl_band_std', 'ctrl_band_p95']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in sig_rows:
        row['ctrl_band_mean'] = ctrl_band_mean
        row['ctrl_band_std'] = ctrl_band_std
        row['ctrl_band_p95'] = ctrl_band_p95
        w.writerow(row)

print(f'fig4b_process: wrote {len(sig_rows)} significant SNHG5 targets '
      f'(of {len(rows)} total); control Cohen\'s d band = '
      f'{ctrl_band_mean:.3f} ± {ctrl_band_std:.3f}, p95 = {ctrl_band_p95:.3f}')
