# panel_h_prescreen_scatter.py — Transcriptomics prescreening scatter (mean vs var, 1:1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * 3.0); FSL = round(6.4 * 3.0)
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'figure.constrained_layout.use': True, 'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, '..', 'output')
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig2h_prescreen_scatter.csv'))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
# All genes (grey, small)
other = df[~df['prescreened']]
ax.scatter(other['mean_expr'], other['var_expr'], s=5, c='#CCCCCC', alpha=0.4, edgecolors='none', label='Filtered out')
# Prescreened (blue, large)
ps = df[df['prescreened']]
ax.scatter(ps['mean_expr'], ps['var_expr'], s=20, c='#0072B2', alpha=0.7, edgecolors='white', linewidths=0.3, label='Prescreened')
ax.set_xlabel('Mean Expression')
ax.set_ylabel('Variance')
ax.legend(frameon=False, loc='upper left', prop={'size': FS-4}, markerscale=2)
ax.grid(linestyle=':', alpha=0.4)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'h.{ext}'), dpi=300)
plt.close(); print('Saved h.png/pdf/svg')
