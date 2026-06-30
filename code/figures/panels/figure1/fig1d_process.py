#!/usr/bin/env python3
"""Processing: fig1d_raw*.csv → fig1d_processed.csv
Merge training logs from 3 omics layers into a single tidy dataframe.
"""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# Read training logs
trans = pd.read_csv(os.path.join(DIR, 'fig1d_raw.csv'))
prot = pd.read_csv(os.path.join(DIR, 'fig1d_raw2.csv'))
meta = pd.read_csv(os.path.join(DIR, 'fig1d_raw3.csv'))

# Add omics label
trans['omics'] = 'Transcriptomics'
prot['omics'] = 'Proteomics'
meta['omics'] = 'Metabolomics'

# Combine
df = pd.concat([trans, prot, meta], ignore_index=True)

# Keep relevant columns: epoch, train_total (total loss), val_total, omics
df = df[['epoch', 'train_total', 'val_total', 'omics']].copy()
df.columns = ['epoch', 'train_loss', 'val_loss', 'omics']

# Normalize losses to [0,1] range per omics for comparison
for omic in df['omics'].unique():
    mask = df['omics'] == omic
    max_val = df.loc[mask, 'val_loss'].max()
    df.loc[mask, 'train_loss_norm'] = df.loc[mask, 'train_loss'] / max_val
    df.loc[mask, 'val_loss_norm'] = df.loc[mask, 'val_loss'] / max_val

df.to_csv(os.path.join(DIR, 'fig1d_processed.csv'), index=False)
print(f"Processed: {len(df)} rows, omics: {df['omics'].unique().tolist()}")
