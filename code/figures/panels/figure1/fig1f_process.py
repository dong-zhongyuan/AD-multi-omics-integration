#!/usr/bin/env python3
"""Processing: fig1f_raw.csv → fig1f_processed.csv
Normalize 3-omics network metrics to [0,1] per metric for radar chart.
"""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig1f_raw.csv'))

metrics = ['n_edges', 'mean_strength', 'mean_confidence_stability',
           'mean_confidence_snr', 'mean_confidence_consistency']

# Normalize each metric to [0,1] across omics
for m in metrics:
    mx = df[m].max()
    if mx > 0:
        df[f'{m}_norm'] = df[m] / mx

df.to_csv(os.path.join(DIR, 'fig1f_processed.csv'), index=False)
print(f"Processed: {len(df)} rows, metrics normalized")
