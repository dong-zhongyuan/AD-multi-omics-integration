#!/usr/bin/env python3
"""Plot: fig7b_processed.csv → fig7b.svg/pdf/tiff
Cross-tissue DEG consistency — scatter of iNPH vs AD log fold-change.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig7b_processed.csv'))

fig, ax = plt.subplots(figsize=(20, 10))

# Resolve the AD logFC column name once (source files vary in casing)
y_col = next((c for c in ['LogFC', 'logFC', 'logfc'] if c in df.columns), df.columns[-1])

# Color by consistency
consistent = df[df['consistent'] == True]
inconsistent = df[df['consistent'] == False]

ax.scatter(inconsistent['brain_logfc'], inconsistent[y_col],
           facecolors='none', edgecolors=C_NONSIG, s=30, linewidths=1.2,
           alpha=0.6, zorder=2, label='Inconsistent')
ax.scatter(consistent['brain_logfc'], consistent[y_col],
           facecolors='none', edgecolors=C_TRANSCRIPTOMICS, s=30, linewidths=1.2,
           alpha=0.6, zorder=3, label='Consistent')

# Edge genes
edge = df[df['is_edge_gene'] == True]
if len(edge) > 0:
    ax.scatter(edge['brain_logfc'], edge[y_col],
               facecolors='none', edgecolors=C_SIGNIFICANT, s=80, linewidths=2.5,
               marker='D', zorder=5, label='Edge genes')

ax.axhline(0, color='grey', linewidth=1.0, linestyle='--', zorder=0)
ax.axvline(0, color='grey', linewidth=1.0, linestyle='--', zorder=0)
ax.set_xlabel('iNPH brain logFC')
ax.set_ylabel('AD brain logFC')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
clean(ax)
fig.subplots_adjust(left=0.12, right=0.78, bottom=0.16, top=0.98)
save(fig, OUT_DIR, 'fig7b')
