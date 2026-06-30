import os
#!/usr/bin/env python3
"""
准备受试者级 PLASMA 蛋白诊断分析数据

说明：
- 动态读取 step5 当前的蛋白诊断候选
- 仅保留 baseline (`VISCODE == bl`) 且 QC 通过的样本
- 对同一受试者的重复记录先按蛋白取均值，再透视为受试者级矩阵
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[4])

def prepare_plasma_diagnostic_data():
    print("=" * 80)
    print("准备PLASMA蛋白诊断效力分析数据")
    print("=" * 80)
    
    # 0. 读取诊断靶点基因列表
    diagnostic_file = PROJECT_ROOT / 'output/step5_gene_classification/diagnostic_targets.csv'
    diag_df = pd.read_csv(diagnostic_file)
    # 只取蛋白组的基因
    prot_genes = diag_df[diag_df['omics'] == 'proteomics']['gene'].unique().tolist()
    print(f"\n蛋白组诊断靶点: {prot_genes}")
    
    # 1. 读取PLASMA数据
    print("\n1. 读取PLASMA数据...")
    plasma_df = pd.read_csv(PROJECT_ROOT / 'data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv', 
                            low_memory=False)
    
    # 查看有哪些蛋白可用
    available_targets = plasma_df['Target'].unique()
    valid_genes = [g for g in prot_genes if g in available_targets]
    print(f"   NULISA数据中可用的蛋白: {valid_genes}")
    
    if not valid_genes:
        print("⚠️ 没有可用的蛋白数据")
        return None
    
    # 2. 读取诊断数据
    print("\n2. 读取诊断数据...")
    dx_df = pd.read_csv(PROJECT_ROOT / 'data/blood-transcription-protein/DXSUM_17Apr2026.csv')
    dx_baseline = dx_df[dx_df['VISCODE'] == 'bl'].copy()
    dx_baseline['diagnosis_label'] = dx_baseline['DIAGNOSIS'].map({1: 'CN', 2: 'MCI', 3: 'AD'})
    print(f"   基线诊断样本: {len(dx_baseline)}")
    
    # 3. 构建受试者级蛋白矩阵
    print("\n3. 构建受试者级蛋白矩阵...")

    protein_long = plasma_df[
        (plasma_df['SampleMatrixType'] == 'PLASMA') &
        (plasma_df['Target'].isin(valid_genes)) &
        (plasma_df['SampleQC'] == 'passed') &
        (plasma_df['RID'].notna()) &
        (plasma_df['VISCODE'] == 'bl')
    ][['RID', 'Target', 'NPQ']].copy()

    protein_long['RID'] = pd.to_numeric(protein_long['RID'], errors='coerce')
    protein_long['NPQ'] = pd.to_numeric(protein_long['NPQ'], errors='coerce')
    protein_long = protein_long.dropna(subset=['RID', 'NPQ'])

    for gene in valid_genes:
        n_rid = protein_long.loc[protein_long['Target'] == gene, 'RID'].nunique()
        print(f"   {gene}: {n_rid} 个唯一受试者")

    merged_proteins = protein_long.pivot_table(
        index='RID',
        columns='Target',
        values='NPQ',
        aggfunc='mean'
    ).reset_index()

    print(f"   透视后: {len(merged_proteins)} 个唯一受试者")
    
    # 4. 合并诊断标签
    print("\n4. 合并诊断标签...")
    merged = merged_proteins.merge(
        dx_baseline[['RID', 'DIAGNOSIS', 'diagnosis_label']], 
        on='RID', how='inner'
    )
    print(f"   成功匹配: {len(merged)} 个唯一受试者")
    print(f"   诊断分布:")
    print(merged['diagnosis_label'].value_counts())
    
    # 5. 创建二分类标签（CN vs AD）
    print("\n5. 创建二分类标签（CN vs AD）...")
    merged_binary = merged[merged['diagnosis_label'].isin(['CN', 'AD'])].copy()
    merged_binary['diagnosis'] = (merged_binary['diagnosis_label'] == 'AD').astype(int)
    print(f"   CN: {(merged_binary['diagnosis'] == 0).sum()}, AD: {(merged_binary['diagnosis'] == 1).sum()}")
    
    # 6. 保存数据
    output_file = PROJECT_ROOT / 'data/pbmc_expression_with_diagnosis.csv'
    final_data = merged_binary[['RID'] + valid_genes + ['diagnosis']].copy()
    final_data.to_csv(output_file, index=False)
    print(f"\n✅ 诊断数据已保存: {output_file}")
    print(f"   列: RID, {', '.join(valid_genes)}, diagnosis")
    
    return final_data

if __name__ == '__main__':
    prepare_plasma_diagnostic_data()
