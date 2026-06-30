#!/usr/bin/env python3
"""Processing: fig2a_raw*.csv → fig2a_processed.csv
Summarize cross-tissue edge distributions for confidence violin plot.
"""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

dfs = []
for omic, suffix in [('Transcriptomics', ''), ('Proteomics', '2'), ('Metabolomics', '3')]:
    d = pd.read_csv(os.path.join(DIR, f'fig2a_raw{suffix}.csv'))
    d['omics'] = omic
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
# Keep relevant columns
cols_keep = ['source', 'target', 'weight', 'confidence_stability', 
             'confidence_snr', 'confidence_consistency', 'omics']
df = df[[c for c in cols_keep if c in df.columns]]
df.to_csv(os.path.join(DIR, 'fig2a_processed.csv'), index=False)
print(f"Processed: {len(df)} edges across 3 omics")
