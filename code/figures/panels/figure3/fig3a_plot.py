#!/usr/bin/env python3
"""Plot: fig3a_processed.csv → fig3c.svg/pdf/tiff
Hub gene network centrality — composite score horizontal bar chart.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3a_processed.csv'))
df = df.sort_values('composite_score', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = range(len(df))
ax.barh(y_pos, df['composite_score'].values, height=0.55,
        facecolor='none', edgecolor=C_PROTEOMICS, linewidth=2.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(df['gene'].values)
ax.set_xlabel('Composite Centrality Score')
# compress x so bars fill the panel without dead space
ax.set_xlim(0, df['composite_score'].max() * 1.08)
ax.invert_yaxis()
clean(ax)
fig.subplots_adjust(left=0.26, right=0.96, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig3a')
