#!/usr/bin/env python3
"""fig4c: Per-gene validation dot matrix.
Each row = source gene, columns = Forward/Reverse x Transcriptomics/Proteomics.
Dot size = n_tested, color intensity = n_significant.
Only shows genes with at least 1 significant hit.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig4c_processed.csv'))

gene_stats = df.groupby('source_gene').agg(
    total_sig=('n_significant', 'sum'),
    total_tested=('n_tested', 'sum'),
).reset_index()
hit_genes = gene_stats[gene_stats['total_sig'] > 0]['source_gene'].tolist()
df = df[df['source_gene'].isin(hit_genes)].copy()

combos = [
    ('transcriptomics', 'Forward'),
    ('transcriptomics', 'Reverse'),
    ('proteomics', 'Forward'),
    ('proteomics', 'Reverse'),
]
combo_labels = ['Trans.\nFwd', 'Trans.\nRev', 'Prot.\nFwd', 'Prot.\nRev']

genes_sorted = gene_stats[gene_stats['source_gene'].isin(hit_genes)].sort_values('total_sig', ascending=True)['source_gene'].tolist()

n_genes = len(genes_sorted)
n_combos = len(combos)

fig, ax = plt.subplots(figsize=(20, 10))

norm = Normalize(vmin=0, vmax=1.0)
cmap = plt.get_cmap('YlOrRd')

for gi, gene in enumerate(genes_sorted):
    for ci, (om, dr) in enumerate(combos):
        row = df[(df['source_gene'] == gene) & (df['omics'] == om) & (df['direction'] == dr)]
        if len(row) == 0:
            continue
        n_tested = int(row['n_tested'].values[0])
        n_sig = int(row['n_significant'].values[0])
        if n_tested == 0:
            continue

        sig_rate = n_sig / n_tested
        size = np.log1p(n_tested) * 150
        color = cmap(norm(sig_rate)) if n_sig > 0 else '#E0E0E0'

        ax.scatter(ci, gi, s=size, c=[color], edgecolors='#333',
                   linewidths=1.5, zorder=3)

ax.set_yticks(range(n_genes))
ax.set_yticklabels(genes_sorted)
ax.set_xticks(range(n_combos))
ax.set_xticklabels(combo_labels)

ax.set_xlim(-0.6, n_combos - 0.4)
ax.set_ylim(-0.6, n_genes - 0.4)
ax.invert_yaxis()

for gi in range(n_genes):
    ax.axhline(gi, color='#F0F0F0', linewidth=0.8, zorder=0)
for ci in range(n_combos):
    ax.axvline(ci, color='#F0F0F0', linewidth=0.8, zorder=0)

ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar_ax = fig.add_axes((0.85, 0.25, 0.015, 0.5))
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label('Validation Rate')

fig.subplots_adjust(left=0.15, right=0.83, top=0.95, bottom=0.15)
save(fig, OUT_DIR, 'fig4c')
