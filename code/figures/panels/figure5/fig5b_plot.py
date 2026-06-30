#!/usr/bin/env python3
"""fig5b: Feature importance comparison.
Network-Guided (4 feat) vs Full Plasma top-10, side by side diverging bar chart.
No in-figure titles (per style.py convention).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig5b_processed.csv'))

ng = df[df['panel'] == 'network_guided_l1'].sort_values('coef', ascending=True).reset_index(drop=True)
fp = df[df['panel'] == 'full_plasma_l1'].sort_values('coef', key=abs, ascending=True).tail(10).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: Network-Guided (4 features)
y1 = np.arange(len(ng))
colors1 = [C_SIGNIFICANT if c > 0 else C_BLOOD for c in ng['coef']]
ax1.barh(y1, ng['coef'], height=0.5, facecolor='none', edgecolor=colors1, linewidth=2.5)
ax1.scatter(ng['coef'], y1, s=150, facecolors=colors1, edgecolors='white',
            linewidths=2.0, zorder=4)
ax1.set_yticks(y1)
ax1.set_yticklabels(ng['feature'].values)
ax1.set_xlabel('LASSO Coefficient')
ax1.axvline(0, color='gray', linewidth=1.0, alpha=0.5)

# Right: Full Plasma top 10
y2 = np.arange(len(fp))
colors2 = [C_SIGNIFICANT if c > 0 else C_BLOOD for c in fp['coef']]
ax2.barh(y2, fp['coef'], height=0.5, facecolor='none', edgecolor=colors2, linewidth=2.5)
ax2.scatter(fp['coef'], y2, s=150, facecolors=colors2, edgecolors='white',
            linewidths=2.0, zorder=4)
ax2.set_yticks(y2)
ax2.set_yticklabels(fp['feature'].values)
ax2.set_xlabel('LASSO Coefficient')
ax2.axvline(0, color='gray', linewidth=1.0, alpha=0.5)

for ax in [ax1, ax2]:
    clean(ax)

fig.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.22, wspace=0.75)
save(fig, OUT_DIR, 'fig5b')
