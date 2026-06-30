#!/usr/bin/env python3
"""Processing: fig1c_raw*.csv → fig1c_processed.csv
Merge elbow drop analysis from 3 omics layers.
"""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

trans = pd.read_csv(os.path.join(DIR, 'fig1c_raw.csv'))
prot = pd.read_csv(os.path.join(DIR, 'fig1c_raw2.csv'))
meta = pd.read_csv(os.path.join(DIR, 'fig1c_raw3.csv'))

trans['omics'] = 'Transcriptomics'
prot['omics'] = 'Proteomics'
meta['omics'] = 'Metabolomics'

df = pd.concat([trans, prot, meta], ignore_index=True)
df.to_csv(os.path.join(DIR, 'fig1c_processed.csv'), index=False)
print(f"Processed: {len(df)} rows")
