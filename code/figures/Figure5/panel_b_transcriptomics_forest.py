# panel_b_transcriptomics_forest.py — Transcriptomics reverse-KO Cox forest (1:2 landscape).
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h); FSL = round(6.4 * fig_h)
COLOR_SIG, COLOR_NS = '#009E73', '#999999'
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

df = pd.read_csv(os.path.join(DATA, 'fig5b_transcriptomics_forest.csv'))
# balance risk + protective, top-8 each (like Fig 4 panel i)
risk = df[df['HR'] > 1].sort_values('p_cox').head(8)
prot = df[df['HR'] < 1].sort_values('p_cox').head(8)
df = pd.concat([risk, prot]).sort_values('p_cox').iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = ['#D55E00' if (p < 0.05 and hr > 1) else ('#0072B2' if (p < 0.05 and hr < 1) else '#999999')
          for p, hr in zip(df['p_cox'], df['HR'])]
ax.scatter(df['HR'], y, color=colors, s=42, zorder=3, edgecolors='white', linewidths=0.3)
ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, zorder=1)
# annotate drug phase
for i, row in df.iterrows():
    ph = row.get('ot_max_phase', 0)
    if pd.notna(ph) and ph > 0:
        ax.text(row['HR'] + 0.01, i, f'Ph{int(ph)}', va='center', fontsize=6, color='black')
ax.set_yticks(y); ax.set_yticklabels(df['gene'], fontsize=7)
ax.set_xlabel('Hazard ratio (transcriptomics)', fontsize=FSL - 4)
ax.set_xlim(0.84, 1.22); ax.set_ylim(-0.6, len(df) - 0.4)
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'b.{ext}'), dpi=300)
plt.close(fig)
print('Saved b.png/pdf/svg')
