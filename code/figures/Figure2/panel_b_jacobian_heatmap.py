# panel_b_jacobian_heatmap.py — Proteomics Jacobian heatmap (1:1, no gene labels)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * 3.0); FSL = round(6.4 * 3.0)
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': FS,
    'axes.linewidth': 0.8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, '..', 'output')
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig2b_proteomics_jacobian_matrix.csv'), index_col='source')
# Full 20x20 — no label clipping since we hide tick labels

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_position([0.12, 0.12, 0.68, 0.76])

cmap = mcolors.LinearSegmentedColormap.from_list('div', ['#0072B2','#FFFFFF','#D55E00'], N=256)
cmap.set_bad('#F5F5F5')
vmax = np.nanmax(np.abs(df.values))
cax = ax.imshow(df.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal', interpolation='none')

ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.3)
ax.tick_params(which='minor', bottom=False, left=False)
# No tick labels — gene names omitted
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

cax_pos = ax.get_position()
cbar_ax = fig.add_axes([cax_pos.x1 + 0.02, cax_pos.y0, 0.04, cax_pos.height])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS-4); cbar.outline.set_visible(False)
ax.set_xlabel('Target Protein', fontsize=FS-2, labelpad=4)
ax.set_ylabel('Source Protein', fontsize=FS-2, labelpad=4)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'b.{ext}'), dpi=300)
plt.close(); print('Saved b.png/pdf/svg')
