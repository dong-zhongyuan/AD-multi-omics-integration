# panel_h_mapt_km.py — MAPT Kaplan-Meier curve (1:1).
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * fig_h)
FSL = round(6.4 * fig_h)
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
OUT = os.path.join(HERE, '..', 'output')
DATA = os.path.join(HERE, '..', 'data')


def compute_km(df):
    df = df.sort_values('time')
    times, surv, ct, cs = [0], [1.0], [], []
    n_at_risk = len(df)
    cur = 1.0
    for t, g in df.groupby('time'):
        deaths = g['event'].sum()
        censored = len(g) - deaths
        if n_at_risk > 0:
            cur *= (1.0 - deaths / n_at_risk)
        times.append(t); surv.append(cur)
        if censored > 0:
            ct.extend([t] * censored); cs.extend([cur] * censored)
        n_at_risk -= len(g)
    return times, surv, ct, cs


df = pd.read_csv(os.path.join(DATA, 'fig4h_mapt_km.csv'))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

for grp, color in zip(['High MAPT', 'Low MAPT'], [COLOR_HIGH, COLOR_LOW]):
    sub = df[df['group'] == grp]
    t, s, cts, css = compute_km(sub)
    ax.step(t, s, where='post', color=color, linewidth=1.6, label=grp)
    ax.scatter(cts, css, marker='+', color=color, s=22, zorder=3, alpha=0.9)

ax.text(0.05, 0.08, 'log-rank $p$ < 0.001', transform=ax.transAxes, fontsize=FS - 5, weight='bold')
ax.set_xlabel('Time (months)', fontsize=FSL - 4)
ax.set_ylabel('AD-free survival', fontsize=FSL - 4)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=FS - 6, loc='upper right')
ax.tick_params(labelsize=FS - 5)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(os.path.join(OUT, f'h.{ext}'), dpi=300)
plt.close(fig)
print('Saved j.png/pdf/svg')
