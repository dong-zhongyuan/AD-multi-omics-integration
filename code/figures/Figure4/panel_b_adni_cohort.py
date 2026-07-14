# panel_b_adni_cohort.py — ADNI cohort overview (donut, 1:1).
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h)   # 17
FSL = round(6.4 * fig_h)  # 19
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')

df = pd.read_csv(os.path.join(DATA, 'fig4b_adni_cohort.csv'))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

wedges, _, autotexts = ax.pie(
    df['n'], wedgeprops=dict(width=0.42, edgecolor='w'),
    startangle=90, colors=df['color'],
    autopct=lambda p: '{:.0f}'.format(p * df['n'].sum() / 100),
    pctdistance=0.78, textprops=dict(fontsize=FS - 4, color='white', weight='bold'))
ax.legend(wedges, [f"{d} (n={n})" for d, n in zip(df['diagnosis'], df['n'])],
          loc='center', frameon=False, fontsize=FS - 4,
          bbox_to_anchor=(0.5, 0.5))
ax.set_aspect('equal')

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'b.{ext}'), dpi=300)
plt.close(fig)
print('Saved b.png/pdf/svg')
