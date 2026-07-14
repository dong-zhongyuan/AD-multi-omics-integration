# panel_c_training.py — Neural ODE training convergence per omics layer
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

df = pd.read_csv(os.path.join(DATA, 'fig1d_training.csv'))

# Global palette: Transcriptomics=orange, Proteomics=sky blue, Metabolomics=blue-green
omics_colors = {'Transcriptomics': '#0072B2', 'Proteomics': '#D55E00', 'Metabolomics': '#009E73'}
omics_list = ['Transcriptomics', 'Proteomics', 'Metabolomics']

fig, axes = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)

for ax, omics in zip(axes, omics_list):
    sub = df[df['omics'] == omics]
    c = omics_colors[omics]
    ax.plot(sub['epoch'], sub['train_ot'], color=c, linewidth=1.8, linestyle='-', label='Train')
    ax.plot(sub['epoch'], sub['val_ot'], color=c, linewidth=1.8, linestyle='--', label='Val')
    short = {'Transcriptomics': 'Trans', 'Proteomics': 'Prot', 'Metabolomics': 'Met'}[omics]
    ax.set_title(short, color='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('OT Loss' if ax == axes[0] else '')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# Panel letter is added in the PPT assembly, not in the figure.

for ext in ['png', 'pdf', 'svg']:
    plt.savefig(os.path.join(OUT, f'd.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved d.png/pdf/svg')
