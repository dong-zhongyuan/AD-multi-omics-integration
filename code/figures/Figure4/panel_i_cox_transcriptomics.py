# panel_i_cox_transcriptomics.py — Cox HR transcriptomics (forest, dots only, 1:2 landscape).
# Top-15 genes by significance (p_cox); horizontal layout.
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 6.0, 3.0
FS = round(5.6 * fig_h)
FSL = round(6.4 * fig_h)
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

df = pd.read_csv(os.path.join(DATA, 'fig4i_cox_transcriptomics.csv'))
# balance risk (HR>1) and protective (HR<1): take top-N from each side by significance
risk = df[df['HR'] > 1].sort_values('p_cox').head(8)
prot = df[df['HR'] < 1].sort_values('p_cox').head(8)
df = pd.concat([risk, prot]).sort_values('p_cox').reset_index(drop=True)
df = df.iloc[::-1].reset_index(drop=True)   # least-sig at bottom
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = []
for _, row in df.iterrows():
    if row['p_cox'] < 0.05:
        colors.append('#D55E00' if row['HR'] > 1 else '#0072B2')
    else:
        colors.append('#999999')

ax.scatter(df['HR'], y, color=colors, s=50, zorder=3, edgecolors='white', linewidths=0.3)
ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(df['gene'], fontsize=7)
ax.set_xlabel('Hazard ratio', fontsize=FSL - 4)
ax.tick_params(labelsize=FS - 5)
ax.set_ylim(-0.6, len(df) - 0.4)
# tight x-range so the clustered HR 1.08-1.19 points spread out
ax.set_xlim(0.84, 1.22)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'i.{ext}'), dpi=300)
plt.close(fig)
print('Saved i.png/pdf/svg')
