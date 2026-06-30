#!/usr/bin/env python3
"""Plot: fig7a_processed.csv → fig7a.svg/pdf/tiff
External validation AUC — lollipop chart with CI error bars.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig7a_processed.csv'))

# Order: Brain signature is the star, then Full, then Blood
sig_order = ['Brain', 'Blood', 'Full']
tissue_order = ['All_brain', 'BA9', 'Entorhinal_cortex']
tissue_labels = ['All Brain (n=131)', 'BA9 (n=64)', 'Entorhinal (n=62)']
sig_colors = {
    'Brain': C_BRAIN,
    'Blood': C_BLOOD,
    'Full': C_AD,
}
sig_markers = {
    'Brain': 'o',
    'Blood': 's',
    'Full': '^',
}

fig, ax = plt.subplots(figsize=(20, 10))

# Vertical lollipop
y_spacing = np.arange(len(tissue_order))
offsets = {'Brain': -0.2, 'Blood': 0.0, 'Full': 0.2}

for sig in sig_order:
    for j, (t, tl) in enumerate(zip(tissue_order, tissue_labels)):
        row = df[(df['tissue'] == t) & (df['signature'] == sig)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        auc = float(r['auc'])
        lo = float(r['ci_lower'])
        hi = float(r['ci_upper'])
        yp = j + offsets[sig]

        # Stem
        ax.plot([0.5, auc], [yp, yp], color=sig_colors[sig],
                linewidth=2.0, solid_capstyle='round', alpha=0.6)
        # CI error bar
        ax.errorbar(auc, yp, xerr=[[auc - lo], [hi - auc]],
                    fmt='none', ecolor=sig_colors[sig],
                    capsize=5, capthick=2.0, elinewidth=2.0)
        # Marker
        ax.scatter(auc, yp, s=200, facecolors='none',
                   edgecolors=sig_colors[sig], linewidths=2.5,
                   marker=sig_markers[sig], zorder=4)

# Reference line
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

ax.set_yticks(y_spacing)
ax.set_yticklabels(tissue_labels)
ax.set_xlabel('AUC')
ax.set_xlim(0.4, 0.95)
ax.invert_yaxis()

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=C_BRAIN, markersize=12, markeredgewidth=2.5,
           label='Brain Sig. (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor=C_BLOOD, markersize=12, markeredgewidth=2.5,
           label='Blood Sig. (n=21)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
           markeredgecolor=C_AD, markersize=12, markeredgewidth=2.5,
           label='Full Sig. (n=26)'),
]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
          borderaxespad=0.0)

clean(ax)
fig.subplots_adjust(left=0.28, right=0.72, bottom=0.14, top=0.96)
save(fig, OUT_DIR, 'fig7a')
