# panel_e_km_mmp9_cxcr2.py — MMP9 + CXCR2 Kaplan-Meier curves side by side (2:1 landscape).
# Two parallel Phase 3 drug-repurposing targets.
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 6.0, 3.0
FS = round(5.6 * fig_h); FSL = round(6.4 * fig_h)
COLOR_HIGH, COLOR_LOW = '#D55E00', '#0072B2'
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


def compute_km(df):
    df = df.sort_values('time')
    times, surv = [0], [1.0]
    n = len(df); cur = 1.0
    for t, g in df.groupby('time'):
        deaths = g['event'].sum()
        if n > 0: cur *= (1.0 - deaths / n)
        times.append(t); surv.append(cur)
        n -= len(g)
    return times, surv


fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
for ax, gene, fname in [(axes[0], 'MMP9', 'fig5e_mmp9_km.csv'),
                         (axes[1], 'CXCR2', 'fig5i_cxcr2_km.csv')]:
    df = pd.read_csv(os.path.join(DATA, fname))
    for grp, color in zip([f'High {gene}', f'Low {gene}'], [COLOR_HIGH, COLOR_LOW]):
        sub = df[df['group'] == grp]
        t, s = compute_km(sub)
        ax.step(t, s, where='post', color=color, linewidth=1.6, label=grp)
    ax.set_title(gene, fontsize=FS - 4, pad=4)
    ax.set_xlabel('Months from baseline', fontsize=FSL - 4)
    ax.set_ylabel('AD-free probability', fontsize=FSL - 4)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=FS - 7, loc='upper right')
    ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'e.{ext}'), dpi=300)
plt.close(fig)
print('Saved e.png/pdf/svg')
