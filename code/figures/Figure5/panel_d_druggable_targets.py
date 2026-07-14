# panel_d_druggable_targets.py — Druggable target summary (horizontal bar, 1:1).
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h); FSL = round(6.4 * fig_h)
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


def phase_color(ph):
    if pd.isna(ph): return '#999999'
    ph = int(ph)
    if ph == 3: return '#D55E00'
    if ph == 2: return '#E69F00'
    return '#56B4E9'


df = pd.read_csv(os.path.join(DATA, 'fig5d_druggable_targets.csv'))
df = df.sort_values(['ot_max_phase', 'DrugEvidenceScore'], ascending=[False, False]).head(12)
df = df.sort_values('DrugEvidenceScore', ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = df['ot_max_phase'].apply(phase_color)
ax.barh(y, df['DrugEvidenceScore'], color=colors, edgecolor='none', height=0.7)
ax.set_yticks(y); ax.set_yticklabels(df['gene'], fontsize=7)
ax.set_xlabel('Drug evidence score', fontsize=FSL - 4)
ax.set_xlim(0, df['DrugEvidenceScore'].max() * 1.15)
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'd.{ext}'), dpi=300)
plt.close(fig)
print('Saved d.png/pdf/svg')
