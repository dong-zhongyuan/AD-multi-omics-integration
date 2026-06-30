#!/usr/bin/env python3
"""Plot: fig3b_processed.csv → fig3b.svg/pdf/tiff
GenKI virtual-knockout perturbation footprint.

Per knocked-out hub: pale bar = downstream targets tested, solid bar = targets
reaching significance after knockout. The fraction (significant / tested) is the
perturbation's specificity. SNHG5 has the broadest significant footprint
(14/44), establishing it as the lead biological hit explored in fig4b/c.

Replaces the earlier KL-divergence bar chart, which collapsed three proteomics
knockouts to degenerate (±inf) distributions and was dominated by a single
transcriptomic outlier, leaving only 3 interpretable bars.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig3b_processed.csv'))

# already sorted by footprint (desc); plot bottom->top so SNHG5 sits at top
d = df.iloc[::-1].reset_index(drop=True)
y = np.arange(len(d))
genes = d['KO_gene'].values
tested = d['n_target_genes'].values.astype(float)
sig = d['n_significant_targets'].values.astype(float)
is_trans = (d['tissue'] == 'transcriptomics').values
lead = (genes == 'SNHG5')

fig, ax = plt.subplots(figsize=(20, 10))

bar_h = 0.42
tested_c = [C_TRANSCRIPTOMICS if t else C_PROTEOMICS for t in is_trans]
sig_c = [C_SIGNIFICANT if l else c for l, c in zip(lead, tested_c)]

ax.barh(y + bar_h / 2, tested, height=bar_h, facecolor='none',
        edgecolor=tested_c, linewidth=2.8, label='Targets tested')
ax.barh(y - bar_h / 2, sig, height=bar_h, facecolor=sig_c,
        edgecolor='white', linewidth=1.5, label='Targets significant')

for i in range(len(d)):
    ax.text(tested[i] + 0.6, y[i] + bar_h / 2, f'{int(tested[i])}',
            va='center', ha='left', fontsize=F_VALUE, color='#444444')
    if sig[i] > 0:
        ax.text(sig[i] + 0.6, y[i] - bar_h / 2, f'{int(sig[i])}',
                va='center', ha='left', fontsize=F_VALUE, fontweight='bold',
                color=sig_c[i])

ax.set_yticks(y)
ax.set_yticklabels(genes)
ax.set_xlabel('Downstream genes')
ax.set_xlim(0, tested.max() * 1.25)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
          frameon=False, fontsize=F_LEGEND)

clean(ax)
fig.subplots_adjust(left=0.14, right=0.92, top=0.95, bottom=0.20)
save(fig, OUT_DIR, 'fig3b')
