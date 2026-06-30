#!/usr/bin/env python3
"""Plot: fig2d_processed.csv → fig2d.svg/pdf/tiff
Edge counts by directionality — bubble heatmap for 3 omics layers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig2d_processed.csv'))

omics_order = ['transcriptomics', 'proteomics', 'metabolomics']
dir_order = ['Brain-internal', 'Blood-internal', 'Cross-tissue']
omics_labels = ['Transcriptomics', 'Proteomics', 'Metabolomics']
dir_labels = ['Brain-internal', 'Blood-internal', 'Cross-tissue']

# Pivot to matrix
matrix = df.pivot(index='direction', columns='omics', values='edge_count')
matrix = matrix.reindex(index=dir_order, columns=omics_order)

fig, ax = plt.subplots(figsize=(20, 10))

# Bubble sizes proportional to log10(edge_count)
color_map = {
    'transcriptomics': C_TRANSCRIPTOMICS,
    'proteomics': C_PROTEOMICS,
    'metabolomics': C_METABOLOMICS,
}

for i, d in enumerate(dir_order):
    for j, o in enumerate(omics_order):
        val = matrix.loc[d, o]
        size = np.log10(val + 1) * 300  # scale bubble
        ax.scatter(j, i, s=size, facecolors='none', edgecolors=color_map[o],
                   linewidths=3.0, alpha=0.8, zorder=3)
        # Add text annotation
        if val >= 1000:
            label = f'{val/1000:.0f}k'
        else:
            label = f'{val:.0f}'
        ax.text(j, i, label, ha='center', va='center', fontsize=F_VALUE,
                fontweight='bold', color='black', zorder=4)

ax.set_xticks(range(len(omics_order)))
ax.set_xticklabels(omics_labels)
ax.set_yticks(range(len(dir_order)))
ax.set_yticklabels(dir_labels)
ax.set_xlim(-0.7, len(omics_order) - 0.3)
ax.set_ylim(-0.7, len(dir_order) - 0.3)
ax.invert_yaxis()

# Grid
ax.set_axisbelow(True)
ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

clean(ax)
fig.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.96)
save(fig, OUT_DIR, 'fig2d')
