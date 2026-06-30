#!/usr/bin/env python3
"""Plot: fig1b_processed.csv → fig1f.svg/pdf/tiff
Method ablation — OT distance and correlation structure error for 4 methods.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig1b_processed.csv'))

colors_list = [C_TRANSCRIPTOMICS, C_PROTEOMICS, C_METABOLOMICS, C_PURPLE]

fig, axes = plt.subplots(1, 2, figsize=(30, 10))

# Panel left: OT distance
ax = axes[0]
x = np.arange(len(df))
for i, row in df.iterrows():
    ax.bar(i, row['ot_distance_mean'], width=0.55, facecolor='none',
           edgecolor=colors_list[i], linewidth=2.5)
    ax.errorbar(i, row['ot_distance_mean'], yerr=row['ot_distance_sd'],
                fmt='none', ecolor='black', capsize=6, capthick=2.5, elinewidth=2.5)
ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=35, ha='right')
ax.set_ylabel('OT Distance')
clean(ax)

# Panel right: Correlation structure MAE
ax = axes[1]
for i, row in df.iterrows():
    ax.bar(i, row['corr_structure_mae'], width=0.55, facecolor='none',
           edgecolor=colors_list[i], linewidth=2.5)
ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=35, ha='right')
ax.set_ylabel('Corr. MAE')
clean(ax)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.95, wspace=0.25)
save(fig, OUT_DIR, 'fig1b')
