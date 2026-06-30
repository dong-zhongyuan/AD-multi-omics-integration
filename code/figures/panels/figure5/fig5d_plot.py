#!/usr/bin/env python3
"""fig5d: Biomarker disease-correlation ranking with LASSO selection overlay.

Scatter of all 134 candidate plasma biomarkers ranked by absolute disease
correlation (X = rank, Y = |r|). The 4 features retained by the
Network-Guided LASSO are highlighted and labeled — showing the model captures
the strongest disease-associated biomarkers while also picking up KLK6, whose
multivariate contribution is invisible to univariate correlation.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig5d_processed.csv'))

fig, ax = plt.subplots(figsize=(20, 10))

# all biomarkers: grey if not selected, colored if selected
ns = df[~df['selected']]
ax.scatter(ns['rank'], ns['abs_correlation'], s=60,
           facecolors='none', edgecolors=C_NONSIG, linewidths=1.4,
           alpha=0.5, zorder=2)

sel = df[df['selected']]
ax.scatter(sel['rank'], sel['abs_correlation'], s=320,
           facecolors=C_SIGNIFICANT, edgecolors='white', linewidths=2.0,
           zorder=4)
# label selected features — use leader lines so adjacent rank-4/5 points
# don't collide; per-feature text offsets chosen by rank to fan them out
label_xy = {  # gene -> (text_x_offset, text_y) — fanned out so adjacent
              # rank-4 (BD-pTau-231) and rank-5 (pTau-231) labels never collide
    'BD-pTau-231': (-14, 0.030),
    'pTau-231':    ( 14, 0.030),
    'TREM1':       (  0, 0.040),
    'KLK6':        (  0, 0.030),
}
for _, r in sel.iterrows():
    dx, dy = label_xy.get(r['gene'], (0, 0.022))
    px, py = r['rank'], r['abs_correlation']
    tx, ty = px + dx, py + dy
    ax.plot([px, tx], [py, ty], lw=1.6, color=C_SIGNIFICANT,
            zorder=3, clip_on=False)
    ax.annotate(r['gene'], xy=(px, py), xytext=(tx, ty),
                fontsize=F_VALUE, fontweight='bold', color=C_SIGNIFICANT,
                ha='center', va='bottom')

ax.set_xlabel('Biomarker rank (by |disease correlation|)')
ax.set_ylabel('|Pearson r| with diagnosis')
ax.set_xlim(-3, 138)
ax.set_ylim(-0.02, max(df['abs_correlation']) * 1.18)

clean(ax)
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.13)
save(fig, OUT_DIR, 'fig5d')
