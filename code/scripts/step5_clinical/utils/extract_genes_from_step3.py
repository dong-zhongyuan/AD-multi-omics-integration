import os
#!/usr/bin/env python3
"""
从Step3输出提取source和target基因列表
输出基因symbol供Step5各脚本使用
"""

import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, '.')
from tools.gene_id_converter import ensembl_to_symbol

# 项目路径
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[4])
STEP3_OUTPUT = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis/transcriptomics/filtered_cross_tissue_edges.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/step5_clinical_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("从Step3输出提取基因列表")
    print("=" * 80)
    
    # 1. 读取Step3输出
    print(f"\n1. 读取Step3输出: {STEP3_OUTPUT}")
    
    if not STEP3_OUTPUT.exists():
        print(f"❌ 错误: 文件不存在 {STEP3_OUTPUT}")
        sys.exit(1)
    
    edges_df = pd.read_csv(STEP3_OUTPUT)
    print(f"   总边数: {len(edges_df)}")
    print(f"   列名: {list(edges_df.columns)}")
    
    # 2. 提取唯一的source和target基因
    print("\n2. 提取唯一基因...")
    
    source_genes = edges_df['source'].unique()
    target_genes = edges_df['target'].unique()
    all_genes = pd.Series(list(set(source_genes) | set(target_genes)))
    
    print(f"   Source基因数: {len(source_genes)}")
    print(f"   Target基因数: {len(target_genes)}")
    print(f"   总基因数（去重）: {len(all_genes)}")
    
    # 3. 映射ENSG ID到基因symbol
    print("\n3. 映射ENSG ID到基因symbol...")
    mapping = ensembl_to_symbol(all_genes)
    
    # 4. 创建输出数据框
    print("\n4. 生成输出文件...")
    
    # 所有基因列表
    all_genes_df = pd.DataFrame({
        'ensembl_id': list(mapping.keys()),
        'gene_symbol': list(mapping.values())
    })
    all_genes_df = all_genes_df.sort_values('gene_symbol')
    
    # Source基因列表
    source_genes_df = pd.DataFrame({
        'ensembl_id': source_genes,
        'gene_symbol': [mapping[g] for g in source_genes]
    })
    source_genes_df = source_genes_df.sort_values('gene_symbol')
    
    # Target基因列表
    target_genes_df = pd.DataFrame({
        'ensembl_id': target_genes,
        'gene_symbol': [mapping[g] for g in target_genes]
    })
    target_genes_df = target_genes_df.sort_values('gene_symbol')
    
    # 边列表（带基因symbol）
    edges_with_symbols = edges_df.copy()
    edges_with_symbols['source_symbol'] = edges_with_symbols['source'].map(mapping)
    edges_with_symbols['target_symbol'] = edges_with_symbols['target'].map(mapping)
    
    # 重新排列列顺序
    cols = ['source', 'source_symbol', 'target', 'target_symbol'] + \
           [c for c in edges_with_symbols.columns if c not in ['source', 'source_symbol', 'target', 'target_symbol']]
    edges_with_symbols = edges_with_symbols[cols]
    
    # 5. 保存输出
    output_files = {
        'all_genes.csv': all_genes_df,
        'source_genes.csv': source_genes_df,
        'target_genes.csv': target_genes_df,
        'edges_with_symbols.csv': edges_with_symbols
    }
    
    for filename, df in output_files.items():
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"   ✅ {filename}: {len(df)} 行")
    
    # 6. 打印摘要
    print("\n" + "=" * 80)
    print("提取完成")
    print("=" * 80)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"\n基因列表:")
    print(f"  - all_genes.csv: {len(all_genes_df)} 个基因")
    print(f"  - source_genes.csv: {len(source_genes_df)} 个source基因")
    print(f"  - target_genes.csv: {len(target_genes_df)} 个target基因")
    print(f"  - edges_with_symbols.csv: {len(edges_with_symbols)} 条边")
    
    print(f"\nSource基因: {', '.join(source_genes_df['gene_symbol'].tolist())}")
    print(f"\nTarget基因: {', '.join(target_genes_df['gene_symbol'].tolist())}")

if __name__ == "__main__":
    main()
