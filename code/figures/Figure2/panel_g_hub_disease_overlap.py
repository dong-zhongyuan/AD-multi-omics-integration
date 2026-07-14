# panel_g_hub_disease_overlap.py — Hub-disease gene overlap (bar chart, 1:1)
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
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig2g_hub_disease_overlap.csv'))
PALETTE = {'Transcriptomics':'#0072B2','Proteomics':'#D55E00','Metabolomics':'#009E73'}

counts = df.groupby('omics').size().reindex(['Transcriptomics','Proteomics','Metabolomics']).fillna(0)
x = np.arange(len(counts)); colors = [PALETTE[o] for o in counts.index]

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
bars = ax.bar(x, counts.values, color=colors, edgecolor='black', linewidth=1, width=0.6)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, yval+5, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=FS)
ax.set_xticks(x); ax.set_xticklabels(['Trans','Prot','Met'])
ax.set_ylabel('Hub ∩ Disease')
ax.set_yscale('symlog', linthresh=10)
ax.set_yticks([0,10,100,400])
ax.grid(axis='y', linestyle=':', alpha=0.6)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'g.{ext}'), dpi=300)
plt.close(); print('Saved g.png/pdf/svg')
