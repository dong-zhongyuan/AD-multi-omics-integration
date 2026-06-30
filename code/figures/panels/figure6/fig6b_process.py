#!/usr/bin/env python3
"""Processing: fig6b_raw.csv → fig6b_processed.csv
Drug mining ranked — select top 8 by DrugEvidenceScore, normalize evidence columns."""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig6b_raw.csv'), encoding='utf-8-sig')
df = df.sort_values('DrugEvidenceScore', ascending=False).head(8).reset_index(drop=True)

# Normalize evidence columns
for col in ['ot_max_phase', 'chembl_unique_molecules_n', 'dgidb_unique_drugs_n', 'ot_tract_score']:
    if col in df.columns:
        vals = df[col].fillna(0).values
        if vals.max() > 0:
            df[f'{col}_norm'] = np.round(vals / vals.max(), 4)
        else:
            df[f'{col}_norm'] = 0.0

df.to_csv(os.path.join(DIR, 'fig6b_processed.csv'), index=False)
print(f"fig6b_processed: {len(df)} targets, top score={df['DrugEvidenceScore'].max():.2f}")
