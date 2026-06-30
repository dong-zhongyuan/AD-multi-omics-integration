#!/usr/bin/env python3
"""Processing: fig5a_raw.csv → fig5a_processed.csv
Extract diagnostic panel AUC comparison data.
"""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig5a_raw.csv'))
# Keep key columns for visualization
cols = ['panel', 'mode', 'n_features', 'cv_auc_mean', 'cv_auc_sd', 'selected_n']
df_out = df[[c for c in cols if c in df.columns]].copy()
df_out.to_csv(os.path.join(DIR, 'fig5a_processed.csv'), index=False)
print(f"Processed: {len(df_out)} panels")
print(df_out[['panel', 'cv_auc_mean']].to_string())
