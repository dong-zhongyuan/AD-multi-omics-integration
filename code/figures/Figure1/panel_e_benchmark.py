# panel_e_benchmark.py — Fair end-to-end mapping comparison (bar chart, 1:2)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

fig_w, fig_h = 6.0, 3.0
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
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig1e_benchmark.csv'))
PALETTE = {'NeuralODE':'#0072B2','DirectOT':'#999999','Ridge':'#D55E00','Identity':'#CC79A7'}

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
metrics = [('ot_distance_mean','ot_distance_sd','OT Distance'),
           ('mmd_rbf',None,'MMD RBF'),
           ('corr_structure_mae',None,'Structure MAE')]
df['order'] = pd.Categorical(df['method'], ['DirectOT','Identity','Ridge','NeuralODE'])
df = df.sort_values('order')

for ax, (col, sd, title) in zip(axes, metrics):
    colors = [PALETTE[m] for m in df['method']]
    if sd:
        ax.bar(df['method'], df[col], yerr=df[sd], color=colors, capsize=3, edgecolor='black', linewidth=0.5)
    else:
        ax.bar(df['method'], df[col], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title)
    ax.set_xticklabels(df['method'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'e.{ext}'), dpi=300, bbox_inches='tight')
plt.close(); print('Saved e.png/pdf/svg')
