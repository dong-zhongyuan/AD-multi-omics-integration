#!/usr/bin/env python3
"""Processing: fig4d_raw.csv + fig4d_raw2.csv → fig4d_processed.csv
Filter significant negative control targets for SNHG5 and PRKAR2B.
"""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

snhg5 = pd.read_csv(os.path.join(DIR, 'fig4d_raw.csv'))
prkar2b = pd.read_csv(os.path.join(DIR, 'fig4d_raw2.csv'))

# Filter significant and select top
snhg5_sig = snhg5[snhg5['significant'] == True].sort_values('cohens_d', ascending=False).head(4)
prkar2b_sig = prkar2b[prkar2b['significant'] == True].sort_values('cohens_d', ascending=False).head(2)

snhg5_sig['ko_gene'] = 'SNHG5'
prkar2b_sig['ko_gene'] = 'PRKAR2B'

df = pd.concat([snhg5_sig, prkar2b_sig], ignore_index=True)

df.to_csv(os.path.join(DIR, 'fig4d_processed.csv'), index=False)
print(f"Processed: {len(df)} targets (SNHG5={len(snhg5_sig)}, PRKAR2B={len(prkar2b_sig)})")
