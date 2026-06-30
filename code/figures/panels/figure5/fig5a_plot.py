#!/usr/bin/env python3
"""fig5a: Diagnostic panel AUC comparison.
Remove strict_forward and verified_proteomics (weak).
Horizontal bar + error bar style, sorted by AUC.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig5a_processed.csv'))

df = df[~df['panel'].isin(['strict_forward', 'verified_proteomics'])].copy()

labels = {
    'full_plasma_l1': 'Full Plasma (22 feat.)',
    'network_guided_l1': 'Network-Guided (4 feat.)',
}
colors_map = {
    'full_plasma_l1': C_BRAIN,
    'network_guided_l1': C_SIGNIFICANT,
}

df = df.sort_values('cv_auc_mean', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10))

y_pos = np.arange(len(df))
bar_height = 0.45

for i, row in df.iterrows():
    panel = row['panel']
    auc = row['cv_auc_mean']
    sd = row['cv_auc_sd']
    n = int(row['selected_n']) if pd.notna(row['selected_n']) else int(row['n_features'])
    color = colors_map.get(panel, C_BLUE)

    ax.barh(i, auc - 0.5, left=0.5, height=bar_height,
            facecolor='none', edgecolor=color, linewidth=2.5)

    ax.errorbar(auc, i, xerr=sd, fmt='none', ecolor=color,
                capsize=8, capthick=2.5, elinewidth=2.5, zorder=4)

    ax.scatter(auc, i, s=200, facecolors=color, edgecolors='white',
               linewidths=2.0, zorder=5, marker='D')

ax.set_yticks(y_pos)
ax.set_yticklabels([labels.get(p, p) for p in df['panel']])
ax.set_xlabel('AUC (Cross-validation)')
ax.set_xlim(0.45, 1.05)
ax.axvline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.5)

clean(ax)
fig.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.20)
save(fig, OUT_DIR, 'fig5a')
