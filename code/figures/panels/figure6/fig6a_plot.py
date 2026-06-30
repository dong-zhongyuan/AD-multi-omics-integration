#!/usr/bin/env python3
"""Plot: fig6a_processed.csv → fig6a.svg/pdf/tiff
Cox Hazard Ratio Forest Plot — horizontal forest plot with HR and 95% CI.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig6a_processed.csv'))
df = df.sort_values('HR', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 10))

y_pos = range(len(df))
colors = [C_SIGNIFICANT if row['p_value'] < 0.05 else C_NONSIG for _, row in df.iterrows()]

for i, (_, row) in enumerate(df.iterrows()):
    color = colors[i]
    ax.plot([row['HR_lower'], row['HR_upper']], [i, i], color=color,
            linewidth=3.0, solid_capstyle='round')
    ax.scatter([row['HR']], [i], color=color, s=130, zorder=5, marker='D')

ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(df['gene'].values)
ax.set_xlabel('Hazard Ratio')
ax.set_ylabel('')
ax.set_xlim(0.45, 2.05)
clean(ax)

fig.subplots_adjust(left=0.25, right=0.95, bottom=0.14, top=0.97)
save(fig, OUT_DIR, 'fig6a')
