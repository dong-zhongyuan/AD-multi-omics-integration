#!/usr/bin/env python3
"""Processing: fig3c_raw.csv → fig3c_processed.csv
Process Geneformer knockout statistics for visualization.
"""
import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'fig3c_raw.csv'))
# Add -log10(p) equivalent from cross_gene_rank
df['rank_score'] = 1.0 / df['cross_gene_rank']
df = df.sort_values('cross_gene_rank')
df.to_csv(os.path.join(DIR, 'fig3c_processed.csv'), index=False)
print(f"Processed Geneformer: {len(df)} genes")
