#!/usr/bin/env python3
"""Processing: fig1e_raw.csv + fig1e_raw2.csv + fig1e_raw3.csv → fig1e_processed.csv
Use local CSV copies of network statistics for the three omics layers.
"""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

rows = []
for omic, suffix in [('Transcriptomics', ''), ('Proteomics', '2'), ('Metabolomics', '3')]:
    fname = f'fig1e_raw{suffix}.csv'
    stats = pd.read_csv(os.path.join(DIR, fname)).iloc[0].to_dict()
    stats['omics'] = omic
    rows.append(stats)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DIR, 'fig1e_processed.csv'), index=False)
print(f"Processed network stats: {len(df)} omics layers")
print(df.to_string())
