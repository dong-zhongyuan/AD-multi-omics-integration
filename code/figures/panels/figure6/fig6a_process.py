#!/usr/bin/env python3
"""Processing: fig6a_raw.csv → fig6a_processed.csv
Cox univariate results — filter imprecise CIs, sort by HR."""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig6a_raw.csv'))
df['ci_width'] = df['HR_upper'] - df['HR_lower']
df = df[df['ci_width'] <= 1.5].copy()
df = df.sort_values('HR', ascending=True).reset_index(drop=True)

df.to_csv(os.path.join(DIR, 'fig6a_processed.csv'), index=False)
print(f"fig6a_processed: {len(df)} genes after CI width filter")
