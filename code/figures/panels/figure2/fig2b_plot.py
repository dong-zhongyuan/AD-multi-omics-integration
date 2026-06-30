#!/usr/bin/env python3
"""Plot: fig2b_processed.csv → fig2b.svg/pdf/tiff
Top 15 proteomics cross-tissue causal edges — horizontal lollipop by combined
quality score (strength × consistency), colored by AD-biology family.

Replaces the earlier anonymous-Ensembl-ID hub lollipop with named, biologically
readable brain↔blood edges: tau-phosphorylation, amyloid, MAPT, neuropeptide
axes (VEGFD↔NPY, BDNF↔IL7), and TDP-43 / PTN.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig2b_processed.csv'))

# classify each edge into an AD-biology family by its member proteins
def family(edge):
    e = edge.replace(' ↔ ', '|')
    if any(t in e for t in ('pTau', 'BD-pTau', 'MAPT')):
        return 'Tau / MAPT'
    if 'Aβ' in e:
        return 'Amyloid-β'
    if any(t in e for t in ('VEGFD', 'NPY', 'BDNF', 'IL7', 'PTN', 'pTDP43')):
        return 'Neuro-peptide'
    return 'Other'

FAMILY_COLOR = {
    'Tau / MAPT':    C_PROTEOMICS,    # vermillion
    'Amyloid-β':     C_TRANSCRIPTOMICS,  # deep blue
    'Neuro-peptide': C_METABOLOMICS,  # green
    'Other':         '#888888',
}
df['family'] = df['edge'].apply(family)

# plot bottom→top so rank #1 sits at the top
d = df.sort_values('score', ascending=True).reset_index(drop=True)
y = np.arange(len(d))
colors = [FAMILY_COLOR[f] for f in d['family']]

fig, ax = plt.subplots(figsize=(20, 10))

ax.hlines(y, 0, d['score'].values, color=colors, linewidth=3.0, zorder=2,
          alpha=0.85)
ax.scatter(d['score'].values, y, s=160, c=colors, edgecolors='white',
           linewidths=2.0, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(d['edge'].values)
ax.set_xlabel('Cross-tissue edge score')

xmin = 0
xmax = d['score'].max() * 1.10
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.8, len(d) - 0.2)

# family legend (manual, frameless, bottom-right whitespace)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                  markersize=14, label=fam)
           for fam, c in FAMILY_COLOR.items() if fam in set(d['family'])]
ax.legend(handles=handles, loc='lower right', fontsize=26,
          borderaxespad=0.5, handletextpad=0.4)

clean(ax)
fig.subplots_adjust(left=0.34, right=0.96, top=0.95, bottom=0.12)
save(fig, OUT_DIR, 'fig2b')
