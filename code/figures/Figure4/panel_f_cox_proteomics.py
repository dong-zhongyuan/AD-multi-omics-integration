# panel_f_cox_proteomics.py — Cox HR proteomics (forest plot, 1:2 landscape).
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

df = pd.read_csv(os.path.join(DATA, 'fig4f_cox_proteomics.csv'))
df = df.sort_values('p_cox', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

y = np.arange(len(df))
colors = []
for _, row in df.iterrows():
    if row['p_cox'] < 0.05:
        colors.append('#D55E00' if row['HR'] > 1 else '#0072B2')
    else:
        colors.append('#999999')

xerr_lower = df['HR'] - df['HR_lower']
xerr_upper = df['HR_upper'] - df['HR']
ax.errorbar(df['HR'], y, xerr=[xerr_lower, xerr_upper], fmt='none',
            ecolor='#BBBBBB', elinewidth=1.2, capsize=2.5, zorder=2)
ax.scatter(df['HR'], y, color=colors, s=42, zorder=3, edgecolors='white', linewidths=0.3)
ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, zorder=1)
ax.set_yticks(y); ax.set_yticklabels(df['gene'], fontsize=FS - 8)
ax.set_xlabel('Hazard ratio (95% CI)', fontsize=FSL - 4)
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'f.{ext}'), dpi=300)
plt.close(fig)
print('Saved g.png/pdf/svg')
