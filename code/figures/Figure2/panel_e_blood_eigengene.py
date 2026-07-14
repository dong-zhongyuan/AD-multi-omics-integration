# panel_e_blood_eigengene.py — Blood eigengene disease genes per omics (3 horizontal bar subplots, 1:3)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

fig_w, fig_h = 9.0, 3.0
FS = round(5.6 * 3.0); FSL = round(6.4 * 3.0)
mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 0.8,
    'axes.titlesize': FSL, 'axes.titleweight': 'bold',
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS, 'font.size': FS,
    'figure.constrained_layout.use': True, 'axes.spines.top': False, 'axes.spines.right': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, '..', 'output')
df = pd.read_csv(os.path.join(HERE, '..', 'data', 'fig2e_blood_eigengene.csv'))
PALETTE = {'Transcriptomics':'#0072B2','Proteomics':'#D55E00','Metabolomics':'#009E73'}
SHORT = {'Transcriptomics':'Trans','Proteomics':'Prot','Metabolomics':'Met'}
# Force abs_correlation from correlation (handles NaN in abs_correlation column)
df['abs_correlation'] = df['correlation'].abs()

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
for ax, omics in zip(axes, ['Transcriptomics','Proteomics','Metabolomics']):
    sub = df[df['omics']==omics].sort_values('abs_correlation', ascending=True).tail(10)
    ax.barh(sub['gene'], sub['abs_correlation'], color=PALETTE[omics], edgecolor='none', height=0.7)
    ax.set_title(SHORT[omics], color='black')
    ax.set_xlabel('|r|' if ax==axes[1] else '', fontsize=FS)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=FS-4)

for ext in ['png','pdf','svg']:
    plt.savefig(os.path.join(OUT, f'e.{ext}'), dpi=300)
plt.close(); print('Saved e.png/pdf/svg')
