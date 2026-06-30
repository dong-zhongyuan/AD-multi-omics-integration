#!/usr/bin/env python3
"""Plot: fig2c_processed.csv → fig2c.svg/pdf/tiff
Network-scale fingerprint of the three omics layers.

Left panel — input scale: features (plasma) and samples (plasma / CSF) per
omics layer, grouped bars.
Right panel — output scale: number of sensitivity edges recovered in each
network compartment (brain-internal, blood-internal, cross-tissue), log-y so
the 132-protein and 2,000-gene networks are comparable on one axis.

Replaces the earlier 4-panel small-multiples of confidence metrics, which
duplicated fig2a. This panel tells the *scale* story instead.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig2c_processed.csv'))

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}
layers = df['layer'].tolist()
ec = [colors[l] for l in layers]  # per-layer edge colors
x = np.arange(len(layers))
w = 0.26

fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 10),
                                gridspec_kw={'width_ratios': [1, 1.15]})

# shared hatch pattern per series (so the three bars are visually distinct)
HATCH = {'a': '////', 'b': '', 'c': 'xxxx'}

# ── left: input scale (features + samples) ──────────────────────────────
axL.bar(x - w, df['n_features'], width=w, facecolor='none',
        edgecolor=ec, linewidth=2.8, hatch=HATCH['a'], label='Features')
axL.bar(x, df['n_plasma_samples'], width=w, facecolor='none',
        edgecolor=ec, linewidth=2.8, label='Plasma samples')
axL.bar(x + w, df['n_csf_samples'], width=w, facecolor='none',
        edgecolor=ec, linewidth=2.8, hatch=HATCH['c'], label='CSF samples')

for i in range(len(layers)):
    for off, col in [(-w, 'n_features'), (0, 'n_plasma_samples'),
                     (w, 'n_csf_samples')]:
        axL.text(i + off, df[col].iloc[i] * 1.06, f"{int(df[col].iloc[i]):,}",
                 ha='center', va='bottom', fontsize=F_VALUE, color='#333333')

axL.set_yscale('log')
axL.set_xticks(x)
axL.set_xticklabels(['Trans.', 'Prot.', 'Meta.'])
axL.set_ylabel('Count (log scale)')
axL.set_ylim(20, 75000)
axL.legend(loc='upper right', fontsize=F_LEGEND)
clean(axL)

# ── right: output scale — edges per compartment ─────────────────────────
compartments = [
    ('n_brain_edges', 'Brain-internal', -w, HATCH['a']),
    ('n_blood_edges', 'Blood-internal', 0, HATCH['b']),
    ('n_cross_tissue_edges', 'Cross-tissue', w, HATCH['c']),
]
for col, lbl, off, h in compartments:
    axR.bar(x + off, df[col], width=w, facecolor='none',
            edgecolor=ec, linewidth=2.8, hatch=h, label=lbl)
for i in range(len(layers)):
    for col, _, off, _ in compartments:
        v = df[col].iloc[i]
        axR.text(i + off, v * 1.06, f"{int(v):,}", ha='center', va='bottom',
                 fontsize=F_VALUE, color='#333333')

axR.set_yscale('log')
axR.set_xticks(x)
axR.set_xticklabels(['Trans.', 'Prot.', 'Meta.'])
axR.set_ylabel('Sensitivity edges recovered (log scale)')
axR.set_ylim(300, 750000)
axR.legend(loc='upper right', fontsize=F_LEGEND)
clean(axR)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.95, wspace=0.30)
save(fig, OUT_DIR, 'fig2c')
