# panel_e_confidence_scatter.py — GenKI effect vs expression, confidence-encoded.
# 2x2 grid (omics x direction). Each dot = one KO-target pair.
#   x = log1p(target_expression)   — expression magnitude
#   y = cohens_d                   — perturbation effect (signed)
#   marker size = -log10(p_value)  — confidence (bigger = more significant)
#   color  = omics palette
#   alpha  = significant (True=0.85, False=0.18)  — KO-significant targets pop
# Replaces the former GenKI violin: the significant subset rising above the
# low-alpha control cloud shows the same KO-vs-control separation, with the
# added dimensions of effect size and confidence.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LETTER = 'e'
fig_w, fig_h = 6.0, 6.0
FS = round(5.6 * 3.0)       # 17
FSL = round(6.4 * 3.0)      # 19
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')
PALETTE = {'Proteomics': '#D55E00', 'Transcriptomics': '#0072B2'}
SHORT = {'Proteomics': 'Prot', 'Transcriptomics': 'Trans'}

df = pd.read_csv(os.path.join(DATA, 'fig3e_confidence.csv'))

# marker-size encoding (cap to avoid giant dots)
df['nlp'] = (-np.log10(df['p_value'])).clip(upper=15)
SIZE_MIN, SIZE_MAX = 12, 220
df['msize'] = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * (df['nlp'] / df['nlp'].max())

fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=False, sharey=False)
order = [('Proteomics', 'Forward', axes[0, 0]),
         ('Proteomics', 'Reverse', axes[0, 1]),
         ('Transcriptomics', 'Forward', axes[1, 0]),
         ('Transcriptomics', 'Reverse', axes[1, 1])]

for omics, direction, ax in order:
    sub = df[(df['omics'] == omics) & (df['direction'] == direction)]
    nonsig = sub[~sub['significant']]
    sig = sub[sub['significant']]
    color = PALETTE[omics]
    # non-significant first (background cloud), significant on top
    ax.scatter(np.log1p(nonsig['target_expression']), nonsig['cohens_d'],
               s=nonsig['msize'], c=color, alpha=0.15, edgecolors='none')
    ax.scatter(np.log1p(sig['target_expression']), sig['cohens_d'],
               s=sig['msize'], c=color, alpha=0.85,
               edgecolors='white', linewidths=0.4)
    ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--')
    ax.set_xlabel('log$_{10}$(expression + 1)', fontsize=FSL - 2)
    ax.set_ylabel("Cohen's d", fontsize=FSL - 2)
    ax.set_title(f'{SHORT[omics]} {direction[0]}', fontsize=FSL - 2, pad=6)
    ax.tick_params(labelsize=FS - 4)
    ax.grid(linestyle=':', alpha=0.35)
    # add breathing room so large markers / labels don't kiss the axes
    ax.margins(x=0.08, y=0.10)

# generous outer padding so the right column's title/markers aren't clipped
fig.get_layout_engine().set(rect=[0.02, 0.02, 0.96, 0.96],
                            h_pad=0.04, w_pad=0.06)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'{LETTER}.{ext}'), dpi=300)
plt.close(fig)
print(f'Saved {LETTER}.png/pdf/svg')
