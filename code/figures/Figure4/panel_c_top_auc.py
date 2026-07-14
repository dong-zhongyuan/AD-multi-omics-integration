# panel_c_top_auc.py — Top diagnostic biomarker AUC (horizontal bar, 1:2 landscape).
# fig_h = 3 (landscape 6 x 3) so FS/FSL stay at 17/19.
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 6.0, 3.0
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

df = pd.read_csv(os.path.join(DATA, 'fig4c_top_auc.csv'))
df = df.sort_values('auc').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = df['omics'].map({'Proteomics': COLOR_PROT, 'Transcriptomics': COLOR_TRANS})
ax.barh(y, df['auc'], color=colors, height=0.7)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(df['gene'], fontsize=FS - 4)
ax.set_xlabel('Single-gene diagnostic AUC', fontsize=FSL - 2)
ax.set_xlim(0.45, 0.75)
ax.tick_params(labelsize=FS - 4)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'c.{ext}'), dpi=300)
plt.close(fig)
print('Saved c.png/pdf/svg')
