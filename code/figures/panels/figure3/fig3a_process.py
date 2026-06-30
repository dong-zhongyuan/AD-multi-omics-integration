#!/usr/bin/env python3
"""Processing: fig3a_raw.csv → fig3a_processed.csv
Network centrality metrics — passthrough (raw is ready for plotting).
"""
import os, pandas as pd
DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DIR, 'fig3a_raw.csv'))
# Passthrough — raw data is ready for plotting
df.to_csv(os.path.join(DIR, 'fig3a_processed.csv'), index=False)
n = len(df)
print(f"fig3a_processed: {n} rows")
