# panel_j_expression_3d.py — 3D surface heatmap of target expression along aging pseudotime.
# Cartesian coordinates: x = pseudotime (young -> old), y = gene, z = mean z-expression.
# Surface color also encodes z (diverging blue-white-red). Reads the pipeline matrix.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig4j_expression_3d.csv'), index_col=0)
bins = pd.read_csv(os.path.join(DATA, 'fig4j_pseudotime_bins.csv'), index_col=0)

# shift pseudotime mids to 0..1 for a clean x-axis
pt_mids = bins['pt_mid'].values
pt_mids = pt_mids - pt_mids.min()
if pt_mids.max() > 0:
    pt_mids = pt_mids / pt_mids.max()

# order genes by oldest-bin expression descending (up-in-aging at top)
genes_sorted = df.iloc[:, -1].sort_values(ascending=False).index.tolist()
mat = df.loc[genes_sorted]
genes = list(mat.index)
n_g, n_b = mat.shape
AMP = 8.0
Z = mat.values * AMP

X, Y = np.meshgrid(pt_mids, np.arange(n_g), indexing='xy')

mpl.rcParams.update({'font.family': 'Arial', 'axes.linewidth': 0.8, 'pdf.fonttype': 42})

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
# give the 3D axes more room so y-axis gene labels don't collide with the surface
ax.set_box_aspect((3, 5, 2.5))

vmax = np.abs(Z).max()
norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = mpl.colormaps['RdBu_r']

surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)),
                       edgecolor='grey', linewidth=0.2, alpha=0.95,
                       rstride=1, cstride=1, antialiased=True, shade=False)

ax.set_yticks(np.arange(n_g))
ax.set_yticklabels(genes, fontsize=6, verticalalignment='center')
ax.tick_params(axis='y', pad=4)
age_ticks, age_labels = [], []
for age_label in ['3m', '18m', '24m']:
    idxs = np.where(bins['age_mode'].astype(str) == age_label)[0]
    if len(idxs):
        b = idxs[len(idxs) // 2]
        age_ticks.append(pt_mids[b])
        age_labels.append(age_label)
ax.set_xticks(age_ticks)
ax.set_xticklabels(age_labels, fontsize=14, fontweight='bold')
ax.set_xlabel('Aging pseudotime', fontsize=16, labelpad=12, fontweight='bold')
ax.set_ylabel('Gene (target ortholog)', fontsize=16, labelpad=28, fontweight='bold')
ax.set_zlabel('Expression (z-score)', fontsize=16, labelpad=8, fontweight='bold')
ax.view_init(elev=24, azim=-50)
ax.set_zlim(-vmax * 1.1, vmax * 1.1)

m = mpl.cm.ScalarMappable(norm=mpl.colors.TwoSlopeNorm(vmin=-vmax/AMP, vcenter=0, vmax=vmax/AMP), cmap=cmap)
m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.55, pad=0.1, aspect=14)
cb.set_label('Expression (z-score)', fontsize=14, fontweight='bold')
cb.ax.tick_params(labelsize=11)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'j.{ext}'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved h.png/pdf/svg')
