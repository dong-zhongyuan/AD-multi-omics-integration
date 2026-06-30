#!/usr/bin/env python3
"""Processing: fig6c_raw.csv + fig6c_raw2.csv → fig6c_processed.csv
Gene classification flow — merge therapeutic/diagnostic, classify by overlap."""
import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

therapeutic = pd.read_csv(os.path.join(DIR, 'fig6c_raw.csv'))
diagnostic = pd.read_csv(os.path.join(DIR, 'fig6c_raw2.csv'))

ther_genes = therapeutic[therapeutic['endpoint_role'] == 'source'][['omics', 'gene']].drop_duplicates()
ther_genes['classification'] = 'Therapeutic'
diag_genes = diagnostic[diagnostic['endpoint_role'] == 'source'][['omics', 'gene']].drop_duplicates()
diag_genes['classification'] = 'Diagnostic'

all_genes = pd.concat([ther_genes, diag_genes], ignore_index=True)
gene_classes = all_genes.groupby('gene')['classification'].apply(set).reset_index()
gene_classes['final_class'] = gene_classes['classification'].apply(
    lambda x: 'Both' if len(x) > 1 else list(x)[0])

gene_omics = all_genes.groupby('gene')['omics'].first().reset_index()
flow_data = gene_classes.merge(gene_omics, on='gene')
flow_data['omics_label'] = flow_data['omics'].map({
    'transcriptomics': 'Transcriptomics',
    'proteomics': 'Proteomics'
})

flow_data.to_csv(os.path.join(DIR, 'fig6c_processed.csv'), index=False)
print(f"fig6c_processed: {len(flow_data)} genes classified")
print(flow_data[['gene', 'omics_label', 'final_class']].to_string())
