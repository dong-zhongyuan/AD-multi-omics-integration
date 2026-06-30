#!/usr/bin/env python3
"""Plot: fig3d_processed.csv → fig3d.svg/pdf/tiff
Validated cross-tissue edges — effect size vs significance scatter.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3d_processed.csv'))

df['neg_log10_p'] = -np.log10(df['validation_p_value'].clip(lower=1e-10))
sig = df[df['validation_significant'] == True]
ns = df[df['validation_significant'] == False]

fig, ax = plt.subplots(figsize=SZ_SQ)
ax.scatter(ns['validation_effect_size'], ns['neg_log10_p'],
           facecolors='none', edgecolors=C_NONSIG, s=55, linewidths=2.0, marker='o')
ax.scatter(sig['validation_effect_size'], sig['neg_log10_p'],
           facecolors='none', edgecolors=C_SIGNIFICANT, s=55, linewidths=2.0, marker='o')
ax.axhline(-np.log10(0.05), ls='--', color='grey', lw=1.5)
ax.set_xlabel("Cohen's d (effect size)")
ax.set_ylabel(r'$-\log_{10}$(p-value)')
clean(ax)
fig.subplots_adjust(left=0.14, right=0.95, top=0.93, bottom=0.16)
save(fig, OUT_DIR, 'fig3d')
