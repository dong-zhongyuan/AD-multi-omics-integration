#!/usr/bin/env python3
"""Plot: fig3c_processed.csv → fig3a.svg/pdf/tiff
Geneformer cross-gene specificity ranking — horizontal bar chart.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3c_processed.csv'))
df = df.sort_values('rank_score', ascending=False).head(15)

fig, ax = plt.subplots(figsize=SZ_HBAR)
y_pos = range(len(df))
ax.barh(y_pos, df['rank_score'].values, height=0.55,
        facecolor='none', edgecolor=C_TRANSCRIPTOMICS, linewidth=2.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(df['gene'].values)
ax.set_xlabel('Specificity Score')
ax.invert_yaxis()
clean(ax)
fig.subplots_adjust(left=0.28, right=0.95, top=0.93, bottom=0.16)
save(fig, OUT_DIR, 'fig3c')
