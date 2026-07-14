# panel_f_scenic_grid.py — SCENIC regulon perturbation per KO, 1x2 (fwd + rev).
# Horizontal bar of overall_effect x1000; top-N KOs per direction.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LETTER = 'f'
N_TOP = 8
COLOR = '#0072B2'                # Transcriptomics
fig_w, fig_h = 6.0, 3.0          # 2:1 composite
FS = round(5.6 * 3.0)
FSL = round(6.4 * 3.0)
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig3f_scenic.csv'))

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
for ax, direction in zip(axes, ['Forward', 'Reverse']):
    sub = df[df['direction'] == direction].sort_values('overall_effect', ascending=False).head(N_TOP).sort_values('overall_effect', ascending=True)
    y = np.arange(len(sub))
    ax.barh(y, sub['overall_effect'] * 1000.0, color=COLOR, height=0.7, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(sub['KO_gene'], fontsize=FS - 4)
    ax.set_xlabel('Effect (x$10^{-3}$)', fontsize=FSL - 2)
    ax.set_title(direction, fontsize=FSL - 2, pad=4)
    ax.set_ylim(-0.5, len(sub) - 0.5)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=FS - 4)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'{LETTER}.{ext}'), dpi=300)
plt.close(fig)
print(f'Saved {LETTER}.png/pdf/svg')
