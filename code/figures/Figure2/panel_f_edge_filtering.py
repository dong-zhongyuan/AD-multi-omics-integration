# panel_f_edge_filtering.py — Edge count before vs after filtering (grouped bar, 1:2)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
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
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig2f_edge_filtering.csv'))
PALETTE = {'Transcriptomics':'#0072B2','Proteomics':'#D55E00','Metabolomics':'#009E73'}

x = np.arange(len(df)); width = 0.35
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.bar(x - width/2, df['before'], width, label='Before', color='#999999', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, df['after'], width, label='After', color=[PALETTE[o] for o in df['omics']], edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
ax.set_ylabel('Edge Count (log)')
ax.set_xticks(x); ax.set_xticklabels(['Trans','Prot','Met'])
ax.legend(frameon=False, loc='upper right', prop={'size': FS-4})
ax.grid(axis='y', linestyle=':', alpha=0.6)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'f.{ext}'), dpi=300)
plt.close(); print('Saved f.png/pdf/svg')
