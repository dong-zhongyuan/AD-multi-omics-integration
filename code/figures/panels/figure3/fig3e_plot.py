#!/usr/bin/env python3
"""Plot: fig3e_processed.csv → fig3e.svg/pdf/tiff
Edge filtering funnel — step-wise reduction bar chart.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3e_processed.csv'))

# Map color names to actual colors
color_map = {
    'C_TRANSCRIPTOMICS': C_TRANSCRIPTOMICS,
    'C_PROTEOMICS': C_PROTEOMICS,
    'C_METABOLOMICS': C_METABOLOMICS,
}

fig, ax = plt.subplots(figsize=(10, 10))
x_pos = range(len(df))
colors = [color_map.get(c, C_TRANSCRIPTOMICS) for c in df['color']]

for i, (_, row) in enumerate(df.iterrows()):
    ax.bar(i, np.log10(row['count'] + 1), width=0.62,
           facecolor='none', edgecolor=colors[i], linewidth=2.5)

ax.set_xticks(list(x_pos))
labels = [f"Step {int(row['step'])}" for _, row in df.iterrows()]
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel(r'$\log_{10}$(Edges)')
ax.set_xlim(-0.6, len(df) - 0.4)
clean(ax)
fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.20)
save(fig, OUT_DIR, 'fig3e')
