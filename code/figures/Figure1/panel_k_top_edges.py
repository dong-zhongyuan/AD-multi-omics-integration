# panel_k_top_edges.py — Metabolomics top edges (lollipop, 2:1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

LETTER = 'k'; COLOR = '#009E73'
fig_w, fig_h = 6.0, 3.0
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

# Abbreviate metabolite names: take first word / shorten
def short_name(n):
    n = str(n)
    if len(n) <= 12:
        return n
    # take first word up to 12 chars
    parts = n.split()
    return parts[0][:12] if parts else n[:12]

df['edge'] = df['source'].apply(short_name) + '-' + df['target'].apply(short_name)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
y = np.arange(len(df))
ax.hlines(y, 0, df['abs_w'], color=COLOR, alpha=0.5, linewidth=3)
ax.scatter(df['abs_w'], y, color=COLOR, s=40, zorder=3, edgecolors='white', linewidths=0.5)
ax.set_yticks(y); ax.set_yticklabels(df['edge'], fontsize=FS-4)
ax.set_xlabel('Strength', fontsize=FS); ax.grid(axis='x', linestyle=':', alpha=0.6)
ax.set_ylim(-0.5, len(df)-0.5)

# NO bbox_inches='tight' to preserve exact 2:1 aspect ratio
for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'{LETTER}.{ext}'), dpi=300)
plt.close(); print(f'Saved {LETTER}.png/pdf/svg')
