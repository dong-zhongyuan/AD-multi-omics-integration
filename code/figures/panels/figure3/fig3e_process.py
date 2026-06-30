#!/usr/bin/env python3
"""Processing: fig3e_raw.csv → fig3e_processed.csv
Classic funnel data with sequential retention/reduction metrics.
"""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig3e_raw.csv'))
df['retention_pct'] = df['count'] / df['count'].iloc[0] * 100
df['step_reduction_pct'] = 0.0
for i in range(1, len(df)):
    prev = df.loc[i - 1, 'count']
    curr = df.loc[i, 'count']
    df.loc[i, 'step_reduction_pct'] = (1 - curr / prev) * 100

df.to_csv(os.path.join(DIR, 'fig3e_processed.csv'), index=False)
print(f"Processed classic funnel with {len(df)} sequential stages")
