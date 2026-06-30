#!/usr/bin/env python3
"""Plot: fig7c_processed.csv → fig7c.svg/pdf/tiff
AD-signal-dose consistency curve.

X = minimum AD |logFC| decile (1=weakest → 10=strongest); Y = % of genes at or
above that decile whose iNPH (brain) and AD logFC share sign. The curve rises
monotonically from the genome-wide baseline (79.7%, dashed) to 92.9% in the
top decile — showing the concordance is anchored in shared AD biology.

Replaces the earlier volcano, which re-plotted fig7b's quadrant data in a
different geometry. This panel answers a dose-response question instead.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig7c_processed.csv'))

# decile increases left→right by signal strength (weak→strong)
d = df.sort_values('decile').reset_index(drop=True)
x = d['decile'].values
y = d['pct_consistent'].values
overall = y[0]  # decile 1 = all genes

fig, ax = plt.subplots(figsize=(20, 10))

# baseline = genome-wide consistency
ax.axhline(overall, color=C_NONSIG, linewidth=2.5, linestyle='--', zorder=1)
ax.text(0.4, overall + 0.4, f'genome-wide {overall:.1f}%', fontsize=F_ANNOT,
        color='#666666', va='bottom', ha='left')

# 50% chance reference
ax.axhline(50, color='#CCCCCC', linewidth=1.5, linestyle=':', zorder=0)

# curve + markers
ax.plot(x, y, color=C_AD, linewidth=4.0, zorder=3, solid_capstyle='round')
ax.scatter(x, y, s=220, facecolors=C_AD, edgecolors='white',
           linewidths=2.5, zorder=4)

# value labels
for xi, yi in zip(x, y):
    ax.annotate(f'{yi:.1f}%', xy=(xi, yi), xytext=(0, 14),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=F_VALUE, fontweight='bold', color=C_AD)

ax.set_xlabel('AD DEG signal decile  (weak → strong)')
ax.set_ylabel('Direction concordance with iNPH (%)')
ax.set_xticks(x)
ax.set_xticklabels([f'D{i}' for i in x])
ax.set_xlim(0.3, 10.7)
ax.set_ylim(45, 100)

clean(ax)
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig7c')
