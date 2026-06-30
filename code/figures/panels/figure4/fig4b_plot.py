#!/usr/bin/env python3
"""fig4b: SNHG5 KO — Cohen's d lollipop chart for 14 significant targets.

Each row = one significant downstream target of the SNHG5 (transcriptomics,
forward) virtual knockout; stem height = Cohen's d (KO target versus
expression-matched control genes). Sorted top → bottom by effect size so the
strongest hits lead. The negative-control null distribution (mean ± 1 SD of
Cohen's d across all 42 SNHG5 targets) is described in the text rather than
overlaid on the figure.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig4b_processed.csv'))

# Largest effect at top
df = df.sort_values('cohens_d', ascending=True).reset_index(drop=True)

band_mean = df['ctrl_band_mean'].iloc[0]
band_std = df['ctrl_band_std'].iloc[0]
band_p95 = df['ctrl_band_p95'].iloc[0]

fig, ax = plt.subplots(figsize=(20, 10))

y = np.arange(len(df))

# Lollipops
ax.hlines(y, 0, df['cohens_d'], color=C_TRANSCRIPTOMICS,
          linewidth=2.4, zorder=3)
ax.scatter(df['cohens_d'], y, s=180, facecolors=C_TRANSCRIPTOMICS,
           edgecolors='white', linewidths=2.0, zorder=4)

ax.set_yticks(y)
ax.set_yticklabels(df['target_gene'].values)
ax.set_xlabel("Cohen's d")
ax.set_ylim(-0.6, len(df) - 0.4)
ax.set_xlim(-6, 6)

clean(ax)
fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig4b')
