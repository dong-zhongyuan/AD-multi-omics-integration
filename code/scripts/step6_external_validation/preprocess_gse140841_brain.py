#!/usr/bin/env python3
"""
预处理 GSE140841 brain bulk RNA-seq 数据
- 从 RAW.tar 提取 kallisto abundance 文件
- 汇总 transcript-level TPM 到 gene-level
- 解析 series_matrix 获取 metadata
- 输出: processed-data/step6_external_validation/GSE140841_brain_gene_tpm.csv.gz
        processed-data/step6_external_validation/GSE140841_brain_metadata.csv
"""

from __future__ import annotations

import sys
import tarfile
import gzip
import io
from pathlib import Path
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data/external-validation'
PROCESSED_DIR = PROJECT_ROOT / 'processed-data/step6_external_validation'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Parse series matrix for metadata
# ============================================================================
def parse_series_matrix():
    """Parse GSE140841_series_matrix.txt.gz for sample metadata."""
    print("[1] 解析 series matrix metadata...")
    matrix_file = DATA_DIR / 'GSE140841_series_matrix.txt.gz'

    fields = {}
    with gzip.open(matrix_file, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                fields['acc'] = [x.strip('"') for x in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_title'):
                fields['title'] = [x.strip('"') for x in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals = [x.strip('"') for x in line.strip().split('\t')[1:]]
                # Determine field type from first value
                if vals[0].startswith('rin score:'):
                    fields['rin'] = [x.replace('rin score: ', '') for x in vals]
                elif vals[0].startswith('Sex:'):
                    fields['sex'] = [x.replace('Sex: ', '') for x in vals]
                elif vals[0].startswith('age:'):
                    fields['age'] = [x.replace('age: ', '') for x in vals]
                elif vals[0].startswith('diagnosis:'):
                    fields['diagnosis_raw'] = [x.replace('diagnosis: ', '') for x in vals]
                elif vals[0].startswith('trem2 variant:'):
                    fields['trem2'] = [x.replace('trem2 variant: ', '') for x in vals]
                elif vals[0].startswith('tissue:'):
                    fields['tissue'] = [x.replace('tissue: ', '') for x in vals]

    meta = pd.DataFrame(fields)
    
    # Simplify diagnosis
    def simplify_diagnosis(d):
        d_lower = d.lower()
        if 'control' in d_lower:
            return 'Control'
        elif 'mci' in d_lower:
            return 'MCI'
        elif 'ad' in d_lower:
            return 'AD'
        else:
            return 'Other'
    
    meta['diagnosis'] = meta['diagnosis_raw'].apply(simplify_diagnosis)
    meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
    meta['rin'] = pd.to_numeric(meta['rin'], errors='coerce')
    
    print(f"  样本总数: {len(meta)}")
    print(f"  诊断分布: {meta['diagnosis'].value_counts().to_dict()}")
    print(f"  组织分布: {meta['tissue'].value_counts().to_dict()}")
    
    return meta


# ============================================================================
# Load transcript-to-gene mapping
# ============================================================================
def build_tx2gene_from_abundance(abundance_df):
    """Extract transcript-to-gene mapping from target_id (Ensembl transcript IDs)."""
    # Kallisto target_ids are Ensembl transcript IDs (ENST...)
    # We need to map them to gene symbols
    # Strategy: use the first part of target_id to get gene mapping
    # Since we don't have a GTF, we'll aggregate by gene using a simple approach:
    # Load a mapping from Ensembl transcript to gene symbol
    pass


def load_and_aggregate_abundance(tar_path, sample_accs):
    """Load kallisto abundance files from tar and aggregate to gene-level TPM."""
    print("[2] 加载 kallisto abundance 文件并汇总到 gene level...")
    
    # First, we need transcript-to-gene mapping
    # Try to use pyensembl or a simple approach
    # Since kallisto outputs transcript-level TPM, we sum TPM by gene
    
    # We'll use a simple approach: extract gene name from transcript ID
    # For human, ENST -> ENSG mapping is needed
    # Let's try to get it from the data itself or use a lookup
    
    # First pass: get all transcript IDs from one sample
    print("  [2a] 获取 transcript-to-gene 映射...")
    
    # Try to use the gene_id_converter or build mapping
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Load transcript IDs from first file
    sample_tpms = {}
    transcript_ids = None
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        # Map GSM accession to tar member
        acc_to_member = {}
        for m in members:
            # Format: GSM4188623_1_abundance.tsv.gz
            gsm = m.name.split('_')[0]
            acc_to_member[gsm] = m
        
        print(f"  TAR中样本数: {len(acc_to_member)}")
        
        # Load each sample
        for i, acc in enumerate(sample_accs):
            if acc not in acc_to_member:
                continue
            
            member = acc_to_member[acc]
            f = tar.extractfile(member)
            content = gzip.decompress(f.read()).decode('utf-8')
            
            df = pd.read_csv(io.StringIO(content), sep='\t')
            
            if transcript_ids is None:
                transcript_ids = df['target_id'].values
            
            sample_tpms[acc] = df['tpm'].values
            
            if (i + 1) % 20 == 0:
                print(f"    已加载 {i+1}/{len(sample_accs)} 样本")
    
    print(f"  已加载 {len(sample_tpms)} 样本, {len(transcript_ids)} transcripts")
    
    # Build transcript-level matrix
    tpm_matrix = pd.DataFrame(sample_tpms, index=transcript_ids)
    
    # Map transcripts to genes
    # Use Ensembl transcript ID prefix to get gene mapping
    # Strategy: try pyensembl, or use a simpler heuristic
    print("  [2b] Transcript-to-gene 汇总...")
    
    # Try to load mapping from biomart or use cached
    mapping_file = PROCESSED_DIR / 'ensembl_tx2gene.csv'
    if mapping_file.exists():
        tx2gene = pd.read_csv(mapping_file)
        tx_to_gene = dict(zip(tx2gene['transcript_id'], tx2gene['gene_symbol']))
    else:
        # Use mygene to batch query
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            
            # Query in batches
            unique_tx = list(set(transcript_ids))
            # Remove version numbers
            tx_clean = [t.split('.')[0] for t in unique_tx]
            
            print(f"    查询 {len(tx_clean)} transcripts (mygene)...")
            results = mg.querymany(tx_clean[:5000], scopes='ensembl.transcript', 
                                   fields='symbol', species='human', verbose=False)
            
            tx_to_gene = {}
            for r in results:
                if 'symbol' in r and 'query' in r:
                    tx_to_gene[r['query']] = r['symbol']
            
            # Save mapping
            tx2gene_df = pd.DataFrame(list(tx_to_gene.items()), columns=['transcript_id', 'gene_symbol'])
            tx2gene_df.to_csv(mapping_file, index=False)
            print(f"    映射成功: {len(tx_to_gene)} transcripts -> genes")
            
        except ImportError:
            print("    mygene不可用，使用简化方法...")
            # Fallback: aggregate by transcript prefix (less accurate but functional)
            # Actually, let's try a different approach - use the est_counts and 
            # just keep transcript-level for now, then map later
            tx_to_gene = {}
    
    if not tx_to_gene:
        # Last resort: try to use gene names from features if available
        # Or just use transcript IDs and note the limitation
        print("    ⚠️  无法获取 transcript-to-gene 映射")
        print("    使用 transcript-level TPM (将在后续步骤中映射)")
        # Save transcript-level
        tpm_matrix.to_csv(PROCESSED_DIR / 'GSE140841_brain_transcript_tpm.csv.gz', compression='gzip')
        return tpm_matrix, False
    
    # Aggregate: sum TPM by gene
    # Map transcript IDs (remove version)
    tx_clean_index = [t.split('.')[0] for t in tpm_matrix.index]
    gene_labels = [tx_to_gene.get(t, None) for t in tx_clean_index]
    
    tpm_matrix['gene_symbol'] = gene_labels
    # Remove unmapped
    tpm_matrix = tpm_matrix[tpm_matrix['gene_symbol'].notna()]
    # Aggregate
    gene_tpm = tpm_matrix.groupby('gene_symbol').sum()
    
    print(f"  Gene-level TPM: {gene_tpm.shape[0]} genes × {gene_tpm.shape[1]} samples")
    
    return gene_tpm, True


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("预处理 GSE140841 Brain Bulk RNA-seq")
    print("=" * 70)
    
    # Parse metadata
    meta = parse_series_matrix()
    meta.to_csv(PROCESSED_DIR / 'GSE140841_brain_metadata.csv', index=False)
    
    # Load and aggregate TPM
    tar_path = DATA_DIR / 'GSE140841_RAW.tar'
    gene_tpm, is_gene_level = load_and_aggregate_abundance(tar_path, meta['acc'].tolist())
    
    if is_gene_level:
        gene_tpm.to_csv(PROCESSED_DIR / 'GSE140841_brain_gene_tpm.csv.gz', compression='gzip')
        print(f"\n✓ Gene-level TPM 保存: {PROCESSED_DIR / 'GSE140841_brain_gene_tpm.csv.gz'}")
    
    print(f"✓ Metadata 保存: {PROCESSED_DIR / 'GSE140841_brain_metadata.csv'}")
    print(f"\n诊断分布:")
    print(meta['diagnosis'].value_counts())
    print(f"\n组织分布:")
    print(meta['tissue'].value_counts())


if __name__ == '__main__':
    main()
