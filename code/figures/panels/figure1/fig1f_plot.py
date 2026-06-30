#!/usr/bin/env python3
"""Plot: fig1f_processed.csv → fig1e.svg/pdf/tiff
Network quality radar — 5 normalized metrics for 3 omics layers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig1f_processed.csv'))

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}

metrics = ['n_edges_norm', 'mean_strength_norm', 'mean_confidence_stability_norm',
           'mean_confidence_snr_norm', 'mean_confidence_consistency_norm']
labels = ['Edge Count', 'Strength', 'Stability', 'SNR', 'Consistency']

N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True))
# Rotate so gap faces right — no label directly at 0°
ax.set_theta_offset(np.pi / N)

for _, row in df.iterrows():
    omic = row['omics']
    values = [row[m] for m in metrics]
    values += values[:1]
    ax.plot(angles, values, color=colors[omic], linewidth=2.5,
            marker='o', markersize=8, markerfacecolor='none',
            markeredgecolor=colors[omic], markeredgewidth=2.0,
            label=omic)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticklabels([])
ax.grid(True, linewidth=1.0, alpha=0.4)

# Legend at figure level — physically separated from polar labels
fig.legend(loc='center left', bbox_to_anchor=(0.78, 0.5),
           borderaxespad=0.0)

fig.subplots_adjust(left=0.02, right=0.72, top=0.96, bottom=0.06)
save(fig, OUT_DIR, 'fig1f')
