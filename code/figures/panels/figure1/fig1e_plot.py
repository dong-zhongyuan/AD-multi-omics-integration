#!/usr/bin/env python3
"""Plot: fig1e_processed.csv → fig1d.svg/pdf/tiff
Network scale — edge counts across network types for 3 omics (grouped bar).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig1e_processed.csv'))

# Parse edge counts from JSON-like strings
data = []
for _, row in df.iterrows():
    omic = row['omics']
    cross = json.loads(row['cross_tissue'].replace("'", '"'))
    blood = json.loads(row['blood_network'].replace("'", '"'))
    brain = json.loads(row['brain_network'].replace("'", '"'))
    data.append({'omics': omic, 'Cross-tissue': cross['n_edges'],
                 'Blood-internal': blood['n_edges'], 'Brain-internal': brain['n_edges']})

pdf = pd.DataFrame(data)

colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
    'Metabolomics': C_METABOLOMICS,
}

fig, ax = plt.subplots(figsize=(20, 10))

networks = ['Cross-tissue', 'Blood-internal', 'Brain-internal']
x = np.arange(len(networks))
width = 0.25

for i, omic in enumerate(['Transcriptomics', 'Proteomics', 'Metabolomics']):
    vals = pdf[pdf['omics'] == omic][networks].values[0]
    bars = ax.bar(x + i * width - width, np.log10(vals + 1), width=width,
                  facecolor='none', edgecolor=colors[omic], linewidth=2.5,
                  label=omic)

ax.set_xticks(x)
ax.set_xticklabels(networks, rotation=20, ha='right')
ax.set_ylabel(r'$\log_{10}$(Edge Count)')
add_legend(ax)
clean(ax)
fig.subplots_adjust(left=0.20, right=0.72, top=0.95, bottom=0.28)
save(fig, OUT_DIR, 'fig1e')
