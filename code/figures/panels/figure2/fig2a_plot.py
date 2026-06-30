#!/usr/bin/env python3
"""Plot: fig2a_processed.csv → fig2a.svg/pdf/tiff
Edge confidence distributions — box plot with jittered scatter, 3 omics × 2 metrics.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig2a_processed.csv'))

# Sample for visualization
np.random.seed(42)
df_sample = df.groupby('omics').apply(
    lambda x: x.sample(min(300, len(x)), random_state=42)
).reset_index(drop=True)

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}
omics_list = ['Transcriptomics', 'Proteomics', 'Metabolomics']

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

metrics = [
    ('confidence_consistency', 'Consistency'),
    ('confidence_stability', 'Stability'),
]

for ax, (col, ylabel) in zip(axes, metrics):
    positions = [0, 1, 2]
    for i, omic in enumerate(omics_list):
        data = df_sample[df_sample['omics'] == omic][col].dropna().values
        color = colors[omic]
        
        # Box plot
        bp = ax.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                        showfliers=False, zorder=2)
        bp['boxes'][0].set_facecolor('none')
        bp['boxes'][0].set_edgecolor(color)
        bp['boxes'][0].set_linewidth(2.5)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(2.5)
        for w in bp['whiskers']:
            w.set_color(color)
            w.set_linewidth(1.5)
        for c in bp['caps']:
            c.set_color(color)
            c.set_linewidth(1.5)
        
        # Jittered scatter
        rng = np.random.default_rng(42 + i)
        x_jit = np.full(len(data), i) + rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(x_jit, data, facecolors='none', edgecolors=color,
                   marker=MARKERS[i], s=20, linewidths=1.0, alpha=0.6, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Trans.', 'Prot.', 'Meta.'])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    clean(ax)

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.95, wspace=0.25)
save(fig, OUT_DIR, 'fig2a')
