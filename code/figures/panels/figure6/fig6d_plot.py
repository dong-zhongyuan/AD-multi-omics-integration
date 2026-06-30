#!/usr/bin/env python3
"""Plot: fig6d_processed.csv → fig6d.svg/pdf/tiff
NHANES Population Biomarker Trends — 2x2 age-stratified biomarker trend grid.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig6d_processed.csv'))

# Restore the intended 2x2 mini-panel so Figure 6 can be assembled as a balanced 2x2 main figure.
biomarkers = [
    ('lymphocyte_mean', 'lymphocyte_sd', 'Lymph. (%)', C_TRANSCRIPTOMICS, 'o'),
    ('crp_mean', 'crp_sd', 'CRP (mg/L)', C_PROTEOMICS, 's'),
    ('wbc_mean', 'wbc_sd', r'WBC (10$^9$/L)', C_METABOLOMICS, 'D'),
    ('hemoglobin_mean', 'hemoglobin_sd', 'Hb (g/dL)', C_PURPLE, '^'),
]

x = np.arange(len(df))
fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharex=True)

for idx, (ax, (mean_col, sd_col, ylabel, color, marker)) in enumerate(zip(axes.flat, biomarkers)):
    means = df[mean_col].values
    sds = df[sd_col].values
    sems = sds / np.sqrt(df['n'].values)

    ax.plot(
        x, means, color=color, linewidth=2.5, marker=marker,
        markersize=11, markerfacecolor='white', markeredgecolor=color,
        markeredgewidth=2.4, zorder=3
    )
    ax.errorbar(
        x, means, yerr=sems, fmt='none', ecolor='black',
        capsize=6, capthick=2.5, elinewidth=2.5, zorder=2
    )

    lower = float(np.nanmin(means - sems))
    upper = float(np.nanmax(means + sems))
    span = max(upper - lower, max(abs(upper), 1.0) * 0.08)
    ax.set_ylim(lower - span * 0.18, upper + span * 0.24)
    ax.set_xlim(-0.05, len(df) - 0.95 + 0.10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if idx < 2:
        ax.tick_params(axis='x', labelbottom=False)
    else:
        ax.set_xticklabels(df['age_group'].values, rotation=28, ha='right')
    ax.tick_params(axis='y', )
    clean(ax)

fig.supxlabel('Age Group', fontweight='bold', y=0.02)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.98, wspace=0.50, hspace=0.62)
save(fig, OUT_DIR, 'fig6d')
