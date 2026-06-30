#!/usr/bin/env python3
"""fig4d: Forest plot of Cohen's d for top validated targets vs negative controls.
Each row = target gene, showing target KL vs control KL with effect size.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig4d_processed.csv'))

df = df[df['significant'] == True].copy()
df = df[np.isfinite(df['target_kl']) & np.isfinite(df['control_kl_mean'])].copy()
df = df.sort_values('cohens_d', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10))

y_pos = np.arange(len(df))
KO_COLORS = {'SNHG5': C_TRANSCRIPTOMICS, 'PRKAR2B': C_PROTEOMICS}

for i, row in df.iterrows():
    ko = row['ko_gene']
    color = KO_COLORS.get(ko, '#333')

    ctrl_mean = row['control_kl_mean']
    ctrl_std = row['control_kl_std']
    ax.barh(i, ctrl_std * 2, left=ctrl_mean - ctrl_std, height=0.5,
            color=color, alpha=0.15, zorder=1)

    ax.plot([ctrl_mean, ctrl_mean], [i - 0.3, i + 0.3], color=color,
            linewidth=2.0, alpha=0.6, zorder=2)

    ax.scatter(row['target_kl'], i, s=250, facecolors=color,
               edgecolors='white', linewidths=2.0, zorder=4, marker='D')

    ax.text(-0.04, i, f"{row['target_gene']}  ({ko}→)",
            ha='right', va='center', transform=ax.get_yaxis_transform(),
            fontweight='bold', color=color, fontsize=F_VALUE)

ax.set_yticks([])
ax.set_xlabel('KL Divergence')
ax.set_xscale('log')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=C_TRANSCRIPTOMICS, alpha=0.15, edgecolor=C_TRANSCRIPTOMICS,
          label='Control range (SNHG5 KO)'),
    Patch(facecolor=C_PROTEOMICS, alpha=0.15, edgecolor=C_PROTEOMICS,
          label='Control range (PRKAR2B KO)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
           markersize=12, label='Target KL'),
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=F_LEGEND)

clean(ax)
fig.subplots_adjust(left=0.22, right=0.94, top=0.95, bottom=0.20)
save(fig, OUT_DIR, 'fig4d')
