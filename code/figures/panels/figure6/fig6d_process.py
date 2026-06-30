#!/usr/bin/env python3
"""Processing: fig6d_raw.csv → fig6d_processed.csv
NHANES biomarker age trends — sort by age group, validate columns."""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig6d_raw.csv'))
df = df.sort_values('age_group').reset_index(drop=True)

# Validate required columns
required = ['crp_mean', 'crp_sd', 'lymphocyte_mean', 'lymphocyte_sd',
            'wbc_mean', 'wbc_sd', 'hemoglobin_mean', 'hemoglobin_sd', 'n']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"WARNING: missing columns: {missing}")

df.to_csv(os.path.join(DIR, 'fig6d_processed.csv'), index=False)
print(f"fig6d_processed: {len(df)} age groups")
