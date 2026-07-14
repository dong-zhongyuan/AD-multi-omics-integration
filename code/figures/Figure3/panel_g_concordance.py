# panel_g_concordance.py — cross-method concordance, 1x2.
# Left : GenKI vs PPI  (proteomics)   — x=log10(KL), y=PPI max effect x1000
# Right: GenKI vs SCENIC (transcriptomics) — x=log10(KL), y=SCENIC effect x1000
# Spearman rho annotated per subplot.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LETTER = 'g'
fig_w, fig_h = 6.0, 3.0
FS = round(5.6 * 3.0)
FSL = round(6.4 * 3.0)
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

ppi = pd.read_csv(os.path.join(DATA, 'fig3g_genki_vs_ppi.csv')).dropna(subset=['ppi_max'])
sce = pd.read_csv(os.path.join(DATA, 'fig3g_genki_vs_scenic.csv'))

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

# Left: GenKI vs PPI (proteomics, vermilion)
ax = axes[0]
ax.scatter(np.log10(ppi['genki_kl'].clip(lower=1e-2)), ppi['ppi_max'] * 1000.0,
           c='#D55E00', s=55, alpha=0.75, edgecolors='white', linewidths=0.5)
rho = ppi[['genki_kl', 'ppi_max']].corr(method='spearman').iloc[0, 1]
ax.set_xlabel('GenKI log$_{10}$(KL)', fontsize=FSL - 2)
ax.set_ylabel('PPI max effect (x$10^{-3}$)', fontsize=FSL - 2)
ax.text(0.04, 0.96, f'rho = {rho:.2f}', transform=ax.transAxes,
        fontsize=FS - 2, va='top', color='#444444')
ax.set_title('Prot', fontsize=FSL - 2, pad=4)
ax.tick_params(labelsize=FS - 4)
ax.grid(linestyle=':', alpha=0.4)

# Right: GenKI vs SCENIC (transcriptomics, blue)
ax = axes[1]
ax.scatter(np.log10(sce['genki_kl'].clip(lower=1e-2)), sce['scenic_effect'] * 1000.0,
           c='#0072B2', s=45, alpha=0.7, edgecolors='white', linewidths=0.5)
rho = sce[['genki_kl', 'scenic_effect']].corr(method='spearman').iloc[0, 1]
ax.set_xlabel('GenKI log$_{10}$(KL)', fontsize=FSL - 2)
ax.set_ylabel('SCENIC effect (x$10^{-3}$)', fontsize=FSL - 2)
ax.text(0.04, 0.96, f'rho = {rho:.2f}', transform=ax.transAxes,
        fontsize=FS - 2, va='top', color='#444444')
ax.set_title('Trans', fontsize=FSL - 2, pad=4)
ax.tick_params(labelsize=FS - 4)
ax.grid(linestyle=':', alpha=0.4)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'{LETTER}.{ext}'), dpi=300)
plt.close(fig)
print(f'Saved {LETTER}.png/pdf/svg')
