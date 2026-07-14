# panel_h_drug_bubble.py — Drug-evidence bubble chart (1:1).
# x = Hazard ratio, y = -log10(p_cox), bubble size = DrugEvidenceScore,
# color = clinical phase (Phase 3 vermilion, Phase 2 orange, Tractable blue, None gray).
# Combines transcriptomics + proteomics targets. Top targets labeled.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 4.0, 4.0
FS = round(5.6 * fig_h); FSL = round(6.4 * fig_h)
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.default': 'bf',
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output'); DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig5h_drug_bubble.csv'))
df = df.dropna(subset=['HR', 'p_cox'])
df = df[df['p_cox'] > 0].copy()
df['nlp'] = -np.log10(df['p_cox'].clip(lower=1e-300))

def phase_color(ph):
    if pd.isna(ph) or ph == 0: return '#BBBBBB'
    ph = int(ph)
    if ph == 3: return '#D55E00'
    if ph == 2: return '#E69F00'
    return '#56B4E9'

df['color'] = df['ot_max_phase'].apply(phase_color)
# bubble size from DrugEvidenceScore (fill 0 → small)
score = df['DrugEvidenceScore'].fillna(0)
SIZE_MIN, SIZE_MAX = 15, 280
df['msize'] = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * (score / max(score.max(), 0.01))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# significance threshold line
ax.axhline(-np.log10(0.05), color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

ax.scatter(df['HR'], df['nlp'], s=df['msize'], c=df['color'], alpha=0.7,
           edgecolors='white', linewidths=0.4)

# label the top-8 by combined significance × drug score
df['priority'] = df['nlp'] * (1 + score)
top = df.nlargest(8, 'priority')
for _, r in top.iterrows():
    ax.annotate(r['gene'], (r['HR'], r['nlp']), fontsize=6, fontweight='bold',
                xytext=(3, 3), textcoords='offset points', color='#1A1A1A')

ax.set_xlabel('Hazard ratio', fontsize=FSL - 4, fontweight='bold')
ax.set_ylabel(r'$-\log_{10}$(Cox $p$)', fontsize=FSL - 4, fontweight='bold')
ax.tick_params(labelsize=FS - 5)

# legend (phase colors) — manual
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#D55E00', markersize=7, label='Phase 3'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E69F00', markersize=7, label='Phase 2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#56B4E9', markersize=7, label='Tractable'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBBBBB', markersize=7, label='No drug'),
]
ax.legend(handles=legend_items, loc='upper left', frameon=False, fontsize=FS - 8,
          bbox_to_anchor=(0.02, 0.98), borderpad=0.5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'h.{ext}'), dpi=300)
plt.close(fig)
print('Saved h.png/pdf/svg')
