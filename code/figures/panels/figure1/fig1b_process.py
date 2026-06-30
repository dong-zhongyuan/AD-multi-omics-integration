#!/usr/bin/env python3
"""Processing: fig1b_raw.csv → fig1b_processed.csv.

Use the local raw copy of the validated ablation result table.
"""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(DIR, 'fig1b_raw.csv'))
name_map = {
    'Identity': 'Identity',
    'NeuralODE_existing': 'Neural ODE',
    'DirectOT_barycentric': 'Direct OT',
    'Ridge_pseudo_aligned': 'Ridge',
}
order = ['Identity', 'NeuralODE_existing', 'DirectOT_barycentric', 'Ridge_pseudo_aligned']
processed = raw.set_index('method').loc[order].reset_index()
processed['label'] = processed['method'].map(name_map)

processed.to_csv(os.path.join(DIR, 'fig1b_processed.csv'), index=False)
processed.to_csv(os.path.join(DIR, 'fig1b.csv'), index=False)
print(f'Processed fair ablation results: {len(processed)} methods')
