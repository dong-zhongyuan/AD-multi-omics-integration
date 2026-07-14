# panel_i_top_edges.py — Transcriptomics top edges (lollipop, 1:1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

LETTER = 'i'; COLOR = '#0072B2'
fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h)
mpl.rcParams.update({
    'font.family': 'Arial',
    'axes.linewidth': 0.8,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'font.size': 17,
    'legend.fontsize': 17,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, '..', 'output')
df = pd.read_csv(os.path.join(HERE, '..', 'data', f'fig1{LETTER}_top_edges.csv'))
df = df.sort_values('abs_w', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
y = np.arange(len(df))
ax.hlines(y, 0, df['abs_w'], color=COLOR, alpha=0.5, linewidth=3)
ax.scatter(df['abs_w'], y, color=COLOR, s=40, zorder=3, edgecolors='white', linewidths=0.5)
ax.set_yticks(y); ax.set_yticklabels(df['edge'], fontsize=FS-2)
ax.set_xlabel('Strength', fontsize=FS); ax.grid(axis='x', linestyle=':', alpha=0.6)
ax.set_ylim(-0.5, len(df)-0.5)
for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'{LETTER}.{ext}'), dpi=300, bbox_inches='tight')
plt.close(); print(f'Saved {LETTER}.png/pdf/svg')
