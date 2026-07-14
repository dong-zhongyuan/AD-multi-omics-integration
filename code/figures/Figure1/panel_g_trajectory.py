# panel_f_trajectory.py — Continuous trajectory interpolation (dual line)
import os
import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv(os.path.join(DATA, 'fig1g_trajectory.csv'))

# Global palette: blood=orange-red, brain=purple (matches reference)
color_blood = '#E8590C'
color_brain = '#7B2D8B'

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

ax.plot(df['t'], df['ot_to_blood'], color=color_blood, marker='o', markersize=4,
        linewidth=1.8, label='to Blood')
ax.plot(df['t'], df['ot_to_brain'], color=color_brain, marker='s', markersize=4,
        linewidth=1.8, label='to Brain')
ax.axvspan(0.1, 0.9, color='#999999', alpha=0.08)

ax.set_xlabel('Transport coordinate t')
ax.set_ylabel('OT Distance')
ax.set_xlim(0, 1)
ax.legend(frameon=False, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.6)

# Panel letter is added in the PPT assembly, not in the figure.

for ext in ['png', 'pdf', 'svg']:
    plt.savefig(os.path.join(OUT, f'g.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved g.png/pdf/svg')
