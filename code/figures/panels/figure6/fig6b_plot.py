#!/usr/bin/env python3
"""Plot: fig6b_processed.csv → fig6b.svg/pdf/tiff
Drug Target Evidence — multi-axis dot matrix summarizing evidence composition.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig6b_processed.csv'))
df = df.sort_values('DrugEvidenceScore', ascending=True).reset_index(drop=True)

metrics = [
    ('score_phase', 'Phase', C_APPROVED),
    ('score_tractability', 'Tract.', C_CLINICAL),
    ('score_dgidb', 'DGIdb', C_ORANGE),
    ('score_chembl', 'ChEMBL', C_PROTEOMICS),
]
y_pos = np.arange(len(df))

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.5], wspace=0.35)
axm = fig.add_subplot(gs[0, 0])
axs = fig.add_subplot(gs[0, 1], sharey=axm)

tier_markers = {
    'EarlyClinicalTarget': 'D',
    'Tractable(Structure/High)': 'o',
    'Tractable(Literature)': 's',
}

for xline in np.arange(len(metrics)):
    axm.axvline(xline, color='0.88', linewidth=1.2, zorder=0)

for idx, (_, row) in enumerate(df.iterrows()):
    marker = tier_markers.get(row['TargetTier'], 'o')
    for xidx, (col, _, color) in enumerate(metrics):
        value = float(row[col])
        size = 110 + 420 * value
        axm.scatter(xidx, idx, s=size, facecolors='white', edgecolors=color,
                    linewidths=2.5, marker=marker, zorder=3)
        if value > 0:
            axm.scatter(xidx, idx, s=size * 0.22, color=color, zorder=4)

axm.set_xlim(-0.65, len(metrics) - 0.35)
axm.set_xticks(np.arange(len(metrics)))
axm.set_xticklabels([label for _, label, _ in metrics], rotation=15, ha='right')
axm.set_yticks(y_pos)
axm.set_yticklabels(df['symbol'].values)
axm.tick_params(axis='y', labelleft=True, )
axm.tick_params(axis='x', )
axm.set_xlabel('Evidence component')
clean(axm)

scores = df['DrugEvidenceScore'].values
axs.hlines(y_pos, 0, scores, color='0.75', linewidth=2.5, zorder=1)
score_colors = [C_APPROVED if tier == 'EarlyClinicalTarget' else C_CLINICAL
                for tier in df['TargetTier']]
axs.scatter(scores, y_pos, s=190, facecolors='white', edgecolors=score_colors,
            linewidths=2.6, marker='o', zorder=3)
for score, ypos in zip(scores, y_pos):
    axs.text(score + 0.02, ypos, f'{score:.2f}', va='center', ha='left',
             fontsize=F_VALUE, color='0.35')

axs.set_xlim(0, max(scores) * 1.50)
axs.set_xticks([0.0, 0.2, 0.4, 0.6])
axs.tick_params(axis='x', )
axs.tick_params(axis='y', labelleft=False, left=False)
axs.set_xlabel('Total score')
clean(axs)
axs.spines['left'].set_visible(False)

fig.subplots_adjust(left=0.16, right=0.95, top=0.93, bottom=0.16)
save(fig, OUT_DIR, 'fig6b')
