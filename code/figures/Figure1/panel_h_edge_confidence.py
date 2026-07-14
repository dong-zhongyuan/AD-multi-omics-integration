# panel_g_edge_confidence.py — Edge confidence distributions per omics layer
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import os

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 19
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 19
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['font.size'] = 17
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT  = os.path.join(HERE, '..', 'output')
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
data_path = os.path.join(DATA, 'fig1h_edge_confidence.csv')

df = pd.read_csv(data_path)

fig, axes = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
PALETTE_OMICS = {'Transcriptomics': '#0072B2', 'Proteomics': '#D55E00', 'Metabolomics': '#009E73'}
SHORT_OMICS = {'Transcriptomics': 'Trans', 'Proteomics': 'Prot', 'Metabolomics': 'Met'}
metrics = [('confidence_stability', 'Stability'),
           ('confidence_snr', 'SNR'),
           ('confidence_consistency', 'Consistency')]

for ax, (col, title) in zip(axes, metrics):
    sns.violinplot(data=df, x='omics', y=col, ax=ax, palette=PALETTE_OMICS, hue='omics', inner="quartile", linewidth=0.8, legend=False)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # shorten x tick labels to fit narrow subplots
    new_labels = [SHORT_OMICS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

# Panel letter is added in the PPT assembly, not in the figure.

for ext in ['png', 'pdf', 'svg']:
    plt.savefig(os.path.join(OUT, f'h.{ext}'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved h.png/pdf/svg')
