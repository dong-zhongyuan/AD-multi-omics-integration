#!/usr/bin/env python3
"""fig4a: Cross-tissue coexpression validation (AD vs Control).

Dumbbell chart for the 7 pipeline-predicted causal edges. Each line connects
the edge's co-expression Pearson r in Control (left dot) vs AD (right dot).
Lines slope upward = the predicted edge is reinforced in disease, validating
the pipeline's causal inference in an independent cohort.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig4a_processed.csv'))

df['edge_label'] = df['source'] + ' → ' + df['target']
# already sorted by r_diff desc in process; plot weakest at top, strongest bottom
df = df.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10))

y = np.arange(len(df))
x_ctrl, x_ad = 0.30, 0.70   # fixed x positions for the two conditions

for i, row in df.iterrows():
    color = C_SIGNIFICANT if row['significant'] else C_NONSIG
    # connecting line
    ax.plot([x_ctrl, x_ad], [i, i], color=color, linewidth=2.0,
            alpha=0.5, zorder=1)
    # control dot
    ax.scatter(x_ctrl, i, s=180, facecolors='white', edgecolors=C_CONTROL,
               linewidths=2.6, zorder=3)
    # AD dot
    ax.scatter(x_ad, i, s=180, facecolors=C_AD,
               edgecolors='white', linewidths=2.0, zorder=4)
    # r values at each dot
    ax.text(x_ctrl - 0.035, i, f"{row['ctrl_r']:.2f}", ha='right', va='center',
            fontsize=F_VALUE, fontweight='bold', color=C_CONTROL)
    ax.text(x_ad + 0.035, i, f"{row['ad_r']:.2f}", ha='left', va='center',
            fontsize=F_VALUE, fontweight='bold', color=C_AD)

ax.set_yticks(y)
ax.set_yticklabels(df['edge_label'].values)
ax.set_xlim(0.12, 0.88)
ax.set_xticks([x_ctrl, x_ad])
ax.set_xticklabels(['Control', 'AD'])
ax.tick_params(axis='x', length=0)
clean(ax)

# legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor=C_CONTROL, markersize=14, markeredgewidth=2.6,
           label='Control r'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_AD,
           markeredgecolor='white', markersize=14, markeredgewidth=2.0,
           label='AD r'),
    Line2D([0], [0], color=C_SIGNIFICANT, linewidth=2.5, alpha=0.7,
           label='Significant (Fisher p<0.05)'),
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

fig.subplots_adjust(left=0.22, right=0.93, top=0.95, bottom=0.20)
save(fig, OUT_DIR, 'fig4a')
