import os
#!/usr/bin/env python3
"""
准备诊断效力分析数据

从GSE312139血液转录组数据和ADNI诊断数据中提取CST3表达和诊断标签
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gzip

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[4])

def load_blood_expression():
    """加载血液转录组数据"""
    print("加载血液转录组数据...")
    
    blood_file = PROJECT_ROOT / "data/blood-transcription-protein/GSE312139_blood.csv.gz"
    
    with gzip.open(blood_file, 'rt') as f:
        expr_df = pd.read_csv(f, index_col=0)
    
    print(f"  表达数据: {expr_df.shape[0]} 基因 × {expr_df.shape[1]} 样本")
    
    # 转置：行=样本，列=基因
    expr_df = expr_df.T
    
    return expr_df

def load_diagnosis():
    """加载诊断数据"""
    print("\n加载诊断数据...")
    
    dx_file = PROJECT_ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv"
    dx_df = pd.read_csv(dx_file)
    
    print(f"  诊断数据: {len(dx_df)} 条记录")
    
    # DIAGNOSIS编码：1=正常, 2=MCI, 3=AD
    # 提取基线诊断
    dx_baseline = dx_df[dx_df['VISCODE'] == 'bl'].copy()
    
    # 创建二分类标签：0=正常, 1=AD (排除MCI)
    dx_baseline['diagnosis'] = dx_baseline['DIAGNOSIS'].apply(
        lambda x: 0 if x == 1 else (1 if x == 3 else np.nan)
    )
    
    # 只保留正常和AD
    dx_baseline = dx_baseline.dropna(subset=['diagnosis'])
    
    print(f"  基线诊断: {len(dx_baseline)} 条")
    print(f"    正常: {sum(dx_baseline['diagnosis'] == 0)}")
    print(f"    AD: {sum(dx_baseline['diagnosis'] == 1)}")
    
    return dx_baseline[['PTID', 'RID', 'diagnosis']]

def check_cst3_availability(expr_df):
    """检查CST3是否在表达数据中"""
    print("\n检查CST3基因...")
    
    # 查找CST3相关的基因ID
    cst3_genes = [col for col in expr_df.columns if 'CST3' in col.upper()]
    
    if len(cst3_genes) > 0:
        print(f"  找到CST3相关基因: {cst3_genes}")
        return cst3_genes[0]
    
    # 尝试通过ENSG ID查找
    # CST3的ENSG ID: ENSG00000101439
    if 'ENSG00000101439' in expr_df.columns:
        print("  找到CST3 (ENSG00000101439)")
        return 'ENSG00000101439'
    
    print("  ⚠️ CST3不在表达数据中")
    print(f"  可用基因示例: {list(expr_df.columns[:10])}")
    
    return None

def prepare_diagnostic_data():
    """准备诊断效力分析数据"""
    
    print("=" * 80)
    print("准备诊断效力分析数据")
    print("=" * 80)
    
    # 1. 加载表达数据
    expr_df = load_blood_expression()
    
    # 2. 加载诊断数据
    dx_df = load_diagnosis()
    
    # 3. 检查CST3
    cst3_gene = check_cst3_availability(expr_df)
    
    if cst3_gene is None:
        print("\n❌ CST3不在血液转录组数据中")
        print("   血液转录组数据可能不包含蛋白质编码基因CST3")
        print("   需要使用蛋白质组数据或其他包含CST3的数据集")
        return None
    
    # 4. 合并数据
    print("\n合并表达和诊断数据...")
    
    # 样本ID匹配（需要根据实际数据调整）
    # GSE312139样本ID格式：1-10-25-21, 2-10-25-21, ...
    # ADNI样本ID格式：011_S_0002, 011_S_0003, ...
    
    print("  ⚠️ 样本ID格式不匹配")
    print(f"  表达数据样本ID示例: {list(expr_df.index[:5])}")
    print(f"  诊断数据样本ID示例: {list(dx_df['PTID'].head())}")
    print("\n  需要样本ID映射文件来匹配GSE312139和ADNI样本")
    
    return None

if __name__ == '__main__':
    prepare_diagnostic_data()
