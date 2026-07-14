# panel_g_l1_coefficients.py — L1 coefficients (diverging lollipop, 1:1).
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h)
FSL = round(6.4 * fig_h)
COLOR_RISK, COLOR_PROTECT = '#D55E00', '#0072B2'
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

df = pd.read_csv(os.path.join(DATA, 'fig4g_l1_coefficients.csv'))
df = df.sort_values('coef', ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = np.where(df['coef'] > 0, COLOR_RISK, COLOR_PROTECT)
ax.hlines(y, xmin=0, xmax=df['coef'], color=colors, alpha=0.7, linewidth=1.4)
ax.scatter(df['coef'], y, color=colors, s=35, zorder=3, edgecolors='white', linewidths=0.3)
ax.axvline(0, color='black', linewidth=0.6, zorder=1)
ax.set_yticks(y); ax.set_yticklabels(df['feature'], fontsize=FS - 7)
ax.set_xlabel('L1 coefficient', fontsize=FSL - 4)
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'g.{ext}'), dpi=300)
plt.close(fig)
print('Saved f.png/pdf/svg')
