#!/usr/bin/env python3
"""fig5c: ROC curves for the 4 diagnostic panels.

Cross-validated ROC curves comparing Strict-Forward, Verified Proteomics,
Network-Guided, and Full Plasma panels. The Network-Guided curve (4 LASSO
features) closely tracks the Full Plasma curve (22 features), both far above
the weaker fixed panels.
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
df = pd.read_csv(os.path.join(OUT_DIR, 'fig5c_processed.csv'))

PANEL_STYLE = {
    'strict_forward':      {'color': '#BDBDBD', 'ls': ':'},
    'verified_proteomics': {'color': '#9E9E9E', 'ls': '--'},
    'network_guided_l1':   {'color': C_SIGNIFICANT, 'ls': '-'},
    'full_plasma_l1':      {'color': C_PROTEOMICS, 'ls': '-'},
}

fig, ax = plt.subplots(figsize=(20, 10))

# diagonal reference
ax.plot([0, 1], [0, 1], color='grey', ls='-', lw=1.2, alpha=0.4, zorder=1)

legend_items = []
for panel in ['strict_forward', 'verified_proteomics', 'network_guided_l1', 'full_plasma_l1']:
    sub = df[df['panel'] == panel].sort_values('fpr')
    if sub.empty:
        continue
    s = PANEL_STYLE.get(panel, {'color': '#888', 'ls': '-'})
    info = sub.iloc[0]
    ax.plot(sub['fpr'], sub['tpr'], color=s['color'], linestyle=s['ls'],
            linewidth=3.0, zorder=3)
    legend_items.append(
        Line2D([0], [0], color=s['color'], linestyle=s['ls'], linewidth=3.0,
               label=f"{info['label']} ({int(info['n_features'])} feat, AUC={info['auc']:.3f})")
    )

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(handles=legend_items, loc='lower right')

clean(ax)
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig5c')
