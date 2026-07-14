# panel_d_panel_auc.py — Panel-level diagnostic AUC (bar with error bars, 1:1).
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h)
FSL = round(6.4 * fig_h)
COLOR_PROT, COLOR_TRANS = '#D55E00', '#0072B2'
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig4d_panel_auc.csv'))
df['auc_sd'] = df['auc_sd'].fillna(0)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

x = np.arange(len(df))
colors = df['omics'].map({'Proteomics': COLOR_PROT, 'Transcriptomics': COLOR_TRANS})
ax.bar(x, df['auc'], color=colors, yerr=df['auc_sd'], capsize=4, width=0.6, alpha=0.9,
       error_kw=dict(ecolor='#666666', lw=1))
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_xticks(x)
short = {'Proteomics': 'Prot', 'Transcriptomics': 'Trans'}
ax.set_xticklabels([f"{short.get(o, o)}\n({m})" for o, m in zip(df['omics'], df['model'])], fontsize=FS - 6)
ax.set_ylabel('Panel AUC', fontsize=FSL - 4)
ax.set_ylim(0.5, 0.8)
ax.tick_params(labelsize=FS - 4)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'd.{ext}'), dpi=300)
plt.close(fig)
print('Saved d.png/pdf/svg')
