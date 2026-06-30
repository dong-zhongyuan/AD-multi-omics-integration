#!/usr/bin/env python3
"""fig3f: Causal prediction rank vs KO validation effect size.

Scatter of the 22 verified cross-tissue causal edges. X = predicted causal rank
(1 = strongest prediction); Y = virtual-KO validation effect size (Cohen's d).
Marker shape encodes causal direction (Forward CSF→PBMC vs Reverse PBMC→CSF);
color encodes the knocked-out hub gene.

Edges retain significant KO effects across the full prediction rank range
(even rank 40-63), supporting the reliability of the pipeline's causal inference.
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
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3f_processed.csv'))

GENE_COLOR = {
    'SNHG5':    C_TRANSCRIPTOMICS,
    'PRKAR2B':  C_PROTEOMICS,
    'FABP3':    C_METABOLOMICS,
    'MAPT':     C_BRAIN,
    'AGRN':     C_ORANGE,
    'CAVIN2':   C_TEAL,
    'LRBA':     C_PURPLE,
    'CYB5R3':   C_GREEN,
    'SURF1':    C_AD,
}

fig, ax = plt.subplots(figsize=(20, 10))

for direction, sub in df.groupby('direction'):
    marker = 'o' if direction == 'Forward' else 's'
    for gene, g2 in sub.groupby('ko_gene'):
        ax.scatter(g2['predicted_rank'], g2['validation_effect_size'],
                   s=160, marker=marker,
                   facecolors='none', edgecolors=GENE_COLOR.get(gene, '#888'),
                   linewidths=2.4, zorder=3)

# Large-effect reference (Cohen's d = 0.8)
ax.axhline(0.8, color='grey', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

ax.set_xlabel('Predicted causal rank (1 = strongest)')
ax.set_ylabel("KO validation effect size (Cohen's d)")

dir_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#555', markersize=14, markeredgewidth=2.4,
           label='Forward (CSF\u2192PBMC)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='#555', markersize=14, markeredgewidth=2.4,
           label='Reverse (PBMC\u2192CSF)'),
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.5,
           label='Large effect (d = 0.8)'),
]
ax.legend(handles=dir_handles, loc='upper right')

clean(ax)
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig3f')
