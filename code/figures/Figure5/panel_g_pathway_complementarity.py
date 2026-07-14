# panel_g_pathway_complementarity.py — Pathway complementarity (diverging bar, 1:1).
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h); FSL = round(6.4 * fig_h)
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
OUT = os.path.join(HERE, '..', 'output'); DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig5g_pathway_complementarity.csv'))
df = df.sort_values('HR', ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = [COLOR_PROT if 'prot' in str(a) else COLOR_TRANS for a in df['axis']]
ax.barh(y, df['HR'] - 1, left=1, color=colors, height=0.6, edgecolor='none')
ax.axvline(1.0, color='black', linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels([f"{g} ({'Prot' if 'prot' in str(a) else 'Trans'})" for g, a in zip(df['gene'], df['axis'])], fontsize=6)
ax.set_xlabel('Hazard ratio', fontsize=FSL - 4)
ax.set_xlim(0.95, df['HR'].max() * 1.08)
ax.spines['left'].set_visible(False)
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'g.{ext}'), dpi=300)
plt.close(fig)
print('Saved g.png/pdf/svg')
