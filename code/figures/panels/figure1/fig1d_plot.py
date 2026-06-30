#!/usr/bin/env python3
"""Plot: fig1d_processed.csv → fig1b.svg/pdf/tiff
Training convergence — validation loss curves for 3 omics layers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig1d_processed.csv'))

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}

omics_order = ['Transcriptomics', 'Proteomics', 'Metabolomics']
fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharex=True)

for ax, omic in zip(axes, omics_order):
    sub = df[df['omics'] == omic].sort_values('epoch')
    ax.plot(sub['epoch'], sub['val_loss_norm'], color=colors[omic], linewidth=2.5)
    ymin = sub['val_loss_norm'].min()
    ymax = sub['val_loss_norm'].max()
    span = max(ymax - ymin, 0.02)
    ax.set_ylim(ymin - span * 0.15, ymax + span * 0.15)
    ax.set_xlim(1, sub['epoch'].max())
    ax.set_xlabel('')
    ax.text(0.05, 0.78, omic, transform=ax.transAxes, color=colors[omic],
            fontsize=F_VALUE, fontweight='bold', ha='left', va='top')
    clean(ax)

axes[0].set_ylabel('Norm. Val. Loss')
axes[1].set_ylabel('')
axes[2].set_ylabel('')
fig.supxlabel('Epoch', fontweight='bold', y=0.01)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.92, wspace=0.28)
save(fig, OUT_DIR, 'fig1d')
