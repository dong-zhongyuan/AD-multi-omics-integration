#!/usr/bin/env python3
"""Plot: fig7d_processed.csv → fig7d.svg/pdf/tiff
Multi-category direction concordance — bar chart with 95% Wilson CIs.

Each bar is a gene stratum; bars are themed by AD-biology relevance
(grey = reference, AD-red = disease signal). The 50% chance line and the
genome-wide baseline (79.7%) anchor the comparison: background non-DEG genes
sit near chance (~56%), AD-significant DEGs climb to 88.9%.

Replaces the earlier 3-bar thin version. CIs make the small-n edge-gene
stratum (n=20) interpretable rather than misleading.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig7d_processed.csv'))

THEME_COLOR = {
    'reference': '#9E9E9E',
    'iNPH':      C_TRANSCRIPTOMICS,
    'AD':        C_AD,
    'joint':     C_METABOLOMICS,
    'edge':      C_PROTEOMICS,
}
colors = [THEME_COLOR[t] for t in df['theme']]

overall = df.loc[df['theme'] == 'reference'].iloc[0]['pct']  # All genes baseline
n_strata = len(df)
x = np.arange(n_strata)

fig, ax = plt.subplots(figsize=(20, 10))

# bars (hollow, themed edge)
bars = ax.bar(x, df['pct'], width=0.62, facecolor='none',
              edgecolor=colors, linewidth=3.0, zorder=3)

# 95% Wilson CI error bars
ax.errorbar(x, df['pct'],
            yerr=[df['pct'] - df['ci_lower'], df['ci_upper'] - df['pct']],
            fmt='none', ecolor='#333333', capsize=8, capthick=2.5,
            elinewidth=2.5, zorder=4)

# value labels with n
for i, row in df.iterrows():
    ax.text(i, row['ci_upper'] + 1.8, f"{row['pct']:.1f}%",
            ha='center', va='bottom', fontsize=F_VALUE, fontweight='bold',
            color=colors[i])
    ax.text(i, 4, f"n={row['n']}", ha='center', va='bottom',
            fontsize=F_ANNOT, color='#555555')

# reference lines
ax.axhline(50, color='#CCCCCC', linewidth=1.8, linestyle=':', zorder=1)
ax.text(n_strata - 0.5, 50.6, 'chance (50%)', fontsize=F_ANNOT, color='#888888',
        ha='right', va='bottom')
ax.axhline(overall, color=C_NONSIG, linewidth=2.2, linestyle='--', zorder=1)
ax.text(-0.45, overall + 0.6, f'genome-wide {overall:.1f}%', fontsize=F_ANNOT,
        color='#666666', va='bottom', ha='left')

ax.set_xticks(x)
ax.set_xticklabels(df['stratum'])
ax.set_ylabel('Direction concordance with iNPH (%)')
ax.set_ylim(0, 105)

clean(ax)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.95)
save(fig, OUT_DIR, 'fig7d')
