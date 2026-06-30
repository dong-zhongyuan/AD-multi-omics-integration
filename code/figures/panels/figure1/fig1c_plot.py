#!/usr/bin/env python3
"""Plot: fig1c_processed.csv → fig1c.svg/pdf/tiff
Elbow detection — score drop rate across rank bins for 3 omics.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig1c_processed.csv'))

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}
markers_map = {
    'Transcriptomics': 'o',
    'Proteomics': 's',
    'Metabolomics': '^',
}

fig, ax = plt.subplots(figsize=(10, 10))

for omic in ['Transcriptomics', 'Proteomics', 'Metabolomics']:
    sub = df[df['omics'] == omic].sort_values('start_rank')
    x = sub['start_rank'].values
    y = sub['mean_score'].values
    ax.plot(x, y, color=colors[omic], linewidth=2.5, marker=markers_map[omic],
            markersize=10, markerfacecolor='none', markeredgewidth=2.0,
            label=omic)

ax.set_xlabel('Hub Rank')
ax.set_ylabel('Eigengene Score')
ax.legend(loc='upper right', frameon=False)

clean(ax)
fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig1c')
