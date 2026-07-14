# panel_f_druggability_tiers.py — Druggability tier distribution (stacked bar, 1:1).
import os
import pandas as pd
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

df = pd.read_csv(os.path.join(DATA, 'fig5f_druggability_tiers.csv'))
pivot = df.groupby(['omics', 'TargetTier']).size().unstack(fill_value=0)
# enforce tier order
tier_order = ['LateClinicalTarget', 'EarlyClinicalTarget', 'Tractable(Structure/High)', 'Unknown']
pivot = pivot.reindex(columns=[t for t in tier_order if t in pivot.columns], fill_value=0)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

colors = ['#D55E00', '#E69F00', '#56B4E9', '#999999'][:len(pivot.columns)]
pivot.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.5, edgecolor='white', linewidth=0.5)
short_idx = ['Prot' if 'Prot' in str(x) else 'Trans' for x in pivot.index]
ax.set_xticklabels(short_idx, rotation=0, fontsize=FS - 4)
ax.set_ylabel('Number of targets', fontsize=FSL - 4)
ax.set_xlabel('')
ax.legend(title='Tier', frameon=False, fontsize=FS - 7, title_fontsize=FS - 7,
          labels=[t.replace('Target', '').replace('(Structure/High)', '\n(Struct)') for t in pivot.columns])
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'f.{ext}'), dpi=300)
plt.close(fig)
print('Saved f.png/pdf/svg')
