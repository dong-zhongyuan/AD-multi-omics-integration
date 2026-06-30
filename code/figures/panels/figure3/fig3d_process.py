#!/usr/bin/env python3
"""Processing: fig3d_raw.csv → fig3d_processed.csv"""
import os, pandas as pd
DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DIR, 'fig3d_raw.csv'))
# Passthrough — raw data is ready for plotting
df.to_csv(os.path.join(DIR, 'fig3d_processed.csv'), index=False)
n = len(df)
print(f"fig3d_processed: {n} rows")
