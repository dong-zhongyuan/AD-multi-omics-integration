# panel_b_umap.py — 5xFAD data overview: UMAP (tissue + genotype)
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8, 'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'font.size': 17,
    'legend.fontsize': 17,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT  = os.path.join(HERE, '..', 'output')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'fig1b_umap.csv'))

# Global palette: matches reference (Brain=purple, Blood=orange-red)
tissue_colors = {'Brain': '#7B2D8B', 'Blood': '#E8590C'}
geno_colors   = {'WT': '#999999', '5xFAD': '#0072B2'}

# Two square UMAPs; b is 1:2 (6 x 3).
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# pack the two square subplots with a small gap
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.12)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.12)

for tissue, color in tissue_colors.items():
    subset = df[df['tissue'] == tissue]
    axes[0].scatter(subset['UMAP1'], subset['UMAP2'], c=color, s=0.8,
                    alpha=0.15, edgecolors='none', rasterized=True, label=tissue)
axes[0].set_title('Tissue')
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[0].set_aspect('equal')
axes[0].legend(frameon=False, loc='upper right', prop={'size': 17}, markerscale=1.5)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

for geno, color in geno_colors.items():
    subset = df[df['predicted_genotype'] == geno]
    axes[1].scatter(subset['UMAP1'], subset['UMAP2'], c=color, s=0.8,
                    alpha=0.15, edgecolors='none', rasterized=True, label=geno)
axes[1].set_title('Predicted Genotype')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].set_aspect('equal')
axes[1].legend(frameon=False, loc='upper right', prop={'size': 17}, markerscale=1.5)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel letter is added in the PPT assembly, not in the figure.

for ext in ['png', 'pdf', 'svg']:
    plt.savefig(os.path.join(OUT, f'b.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved b.png/pdf/svg')
