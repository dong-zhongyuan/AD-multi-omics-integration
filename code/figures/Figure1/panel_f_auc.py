# panel_f_auc.py — Jacobian derivability AUC (bar chart, 1:1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

fig_w, fig_h = 3.0, 3.0
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'font.size': 17,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, '..', 'output')
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig1f_auc.csv'))
PALETTE = {'NeuralODE':'#0072B2','Ridge':'#D55E00','DirectOT':'#999999'}

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
for _, row in df.iterrows():
    method = row['method']; auc = row['auc']
    if pd.isna(auc):
        ax.bar(method, 0.8, color='white', edgecolor=PALETTE[method], hatch='////', linewidth=1)
        ax.text(method, 0.4, 'N/A', ha='center', va='center', fontweight='bold', color=PALETTE[method], fontsize=17)
    else:
        ax.bar(method, auc, color=PALETTE[method], edgecolor='black', linewidth=0.5)
        ax.text(method, auc+0.02, f'{auc:.3f}', ha='center', va='bottom', fontsize=17)
ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
ax.set_ylabel('Knockout AUC')
ax.set_ylim(0, 1.05)
ax.set_xticklabels(df['method'], rotation=45, ha='right')

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'f.{ext}'), dpi=300, bbox_inches='tight')
plt.close(); print('Saved f.png/pdf/svg')
