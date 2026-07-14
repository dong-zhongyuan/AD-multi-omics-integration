# panel_e_roc.py — ROC curves (proteomics + transcriptomics, 1:1).
import os
import json
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

with open(os.path.join(DATA, 'fig4e_roc.json')) as f:
    data = json.load(f)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(data['proteomics']['fpr'], data['proteomics']['tpr'], color=COLOR_PROT, lw=1.8,
        label=f"Prot (AUC={data['proteomics']['auc']:.3f})")
ax.plot(data['transcriptomics']['fpr'], data['transcriptomics']['tpr'], color=COLOR_TRANS, lw=1.8,
        label=f"Trans (AUC={data['transcriptomics']['auc']:.3f})")
ax.plot([0, 1], [0, 1], color='gray', lw=0.8, linestyle='--')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False positive rate', fontsize=FSL - 4)
ax.set_ylabel('True positive rate', fontsize=FSL - 4)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower right', fontsize=FS - 6, frameon=False)
ax.tick_params(labelsize=FS - 4)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'e.{ext}'), dpi=300)
plt.close(fig)
print('Saved e.png/pdf/svg')
