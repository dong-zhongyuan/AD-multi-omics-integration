#!/usr/bin/env python3
"""
Step 0: 数据预处理 - 为 World Model 准备数据（最终版）

重点：先用蛋白组数据验证 World Model，代谢组和转录组后续整合
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# 导入配置管理器
from tools.config_loader import get_config
config = get_config()
import numpy as np
import anndata as ad
import gzip
import warnings
warnings.filterwarnings('ignore')

# 路径配置
DATA_DIR = config.get_path("paths.data_dir")
OUT_DIR = config.get_path("paths.processed_data_dir")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Step 0: 数据预处理 - 为 World Model 准备数据")
print("="*60)

# ============================================================================
# 1. 处理蛋白组数据（血浆 + CSF）- 核心数据
# ============================================================================
print("\n[1/2] 处理蛋白组数据（核心）...")

nulisa_file = DATA_DIR / "blood-transcription-protein" / "BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv"
print(f"读取: {nulisa_file}")

df_protein = pd.read_csv(nulisa_file, low_memory=False)
print(f"  总行数: {len(df_protein)}")

# 分离血浆和 CSF
df_plasma_prot = df_protein[df_protein['SampleMatrixType'] == 'PLASMA'].copy()
df_csf_prot = df_protein[df_protein['SampleMatrixType'] == 'CSF'].copy()

print(f"  血浆蛋白行数: {len(df_plasma_prot)}")
print(f"  CSF蛋白行数: {len(df_csf_prot)}")

# 转换为宽格式（样本 × 蛋白）
def protein_to_wide(df, tissue_name):
    """将长格式蛋白数据转换为宽格式"""
    print(f"\n  处理 {tissue_name} 蛋白组...")
    
    # 使用 PTID + VISCODE 作为样本ID（配对样本）
    df['sample_id'] = df['PTID'].astype(str) + '_' + df['VISCODE'].astype(str)
    
    # 透视表：样本 × 蛋白
    pivot = df.pivot_table(
        index='sample_id',
        columns='Target',
        values='NPQ',
        aggfunc='mean'  # 如果有重复，取平均
    )
    
    print(f"    样本数: {pivot.shape[0]}")
    print(f"    蛋白数: {pivot.shape[1]}")
    print(f"    缺失值比例: {pivot.isna().sum().sum() / pivot.size * 100:.2f}%")
    
    # 填充缺失值（用蛋白的中位数）
    pivot = pivot.fillna(pivot.median())
    
    # 创建 obs (样本信息)
    obs = pd.DataFrame(index=pivot.index)
    obs['tissue'] = tissue_name
    obs['PTID'] = [x.rsplit('_', 1)[0] for x in pivot.index]
    obs['VISCODE'] = [x.rsplit('_', 1)[1] for x in pivot.index]
    obs['dataset'] = tissue_name  # HepaWorld 需要的 batch_key
    obs['time_score'] = 0.0  # 暂时设为0（后续可以用访视时间）
    
    # 创建 var (蛋白信息)
    var = pd.DataFrame(index=pivot.columns)
    var['feature_type'] = 'protein'
    var['gene'] = pivot.columns  # HepaWorld 需要的 gene 列
    
    # 创建 AnnData
    adata = ad.AnnData(
        X=pivot.values.astype(np.float32),
        obs=obs,
        var=var
    )
    
    return adata

adata_plasma_prot = protein_to_wide(df_plasma_prot, 'plasma')
adata_csf_prot = protein_to_wide(df_csf_prot, 'csf')

# ============================================================================
# 2. 找到配对样本（血浆和CSF都有的患者）
# ============================================================================
print("\n[2/2] 识别配对样本...")

plasma_ptids = set(adata_plasma_prot.obs['PTID'])
csf_ptids = set(adata_csf_prot.obs['PTID'])

paired_ptids = plasma_ptids & csf_ptids

print(f"  血浆患者数: {len(plasma_ptids)}")
print(f"  CSF患者数: {len(csf_ptids)}")
print(f"  配对患者数: {len(paired_ptids)}")

# 筛选配对样本
adata_plasma_paired = adata_plasma_prot[adata_plasma_prot.obs['PTID'].isin(paired_ptids)].copy()
adata_csf_paired = adata_csf_prot[adata_csf_prot.obs['PTID'].isin(paired_ptids)].copy()

print(f"\n  配对后血浆样本数: {adata_plasma_paired.n_obs}")
print(f"  配对后CSF样本数: {adata_csf_paired.n_obs}")

# ============================================================================
# 3. 保存处理后的数据
# ============================================================================
print("\n[3/3] 保存处理后的数据...")

# 保存完整蛋白组数据
adata_plasma_prot.write_h5ad(OUT_DIR / "plasma_proteomics_full.h5ad")
print(f"  ✅ plasma_proteomics_full.h5ad ({adata_plasma_prot.n_obs} 样本 × {adata_plasma_prot.n_vars} 蛋白)")

adata_csf_prot.write_h5ad(OUT_DIR / "csf_proteomics_full.h5ad")
print(f"  ✅ csf_proteomics_full.h5ad ({adata_csf_prot.n_obs} 样本 × {adata_csf_prot.n_vars} 蛋白)")

# 保存配对蛋白组数据（用于 World Model）
adata_plasma_paired.write_h5ad(OUT_DIR / "plasma_proteomics_paired.h5ad")
print(f"  ✅ plasma_proteomics_paired.h5ad ({adata_plasma_paired.n_obs} 样本 × {adata_plasma_paired.n_vars} 蛋白)")

adata_csf_paired.write_h5ad(OUT_DIR / "csf_proteomics_paired.h5ad")
print(f"  ✅ csf_proteomics_paired.h5ad ({adata_csf_paired.n_obs} 样本 × {adata_csf_paired.n_vars} 蛋白)")

# ============================================================================
# 4. 数据质量检查
# ============================================================================
print("\n" + "="*60)
print("数据质量检查")
print("="*60)

def check_data_quality(adata, name):
    print(f"\n【{name}】")
    print(f"  样本数: {adata.n_obs}")
    print(f"  特征数: {adata.n_vars}")
    print(f"  数据类型: {adata.X.dtype}")
    print(f"  缺失值: {np.isnan(adata.X).sum()} ({np.isnan(adata.X).sum() / adata.X.size * 100:.2f}%)")
    print(f"  数据范围: [{adata.X.min():.2f}, {adata.X.max():.2f}]")
    print(f"  数据均值: {adata.X.mean():.2f}")
    print(f"  数据标准差: {adata.X.std():.2f}")

check_data_quality(adata_plasma_prot, "血浆蛋白组（完整）")
check_data_quality(adata_csf_prot, "CSF蛋白组（完整）")
check_data_quality(adata_plasma_paired, "血浆蛋白组（配对）")
check_data_quality(adata_csf_paired, "CSF蛋白组（配对）")

# ============================================================================
# 5. 生成数据摘要报告
# ============================================================================
print("\n" + "="*60)
print("数据摘要报告")
print("="*60)

summary = {
    "plasma_full": {
        "samples": adata_plasma_prot.n_obs,
        "features": adata_plasma_prot.n_vars,
        "patients": len(plasma_ptids)
    },
    "csf_full": {
        "samples": adata_csf_prot.n_obs,
        "features": adata_csf_prot.n_vars,
        "patients": len(csf_ptids)
    },
    "paired": {
        "patients": len(paired_ptids),
        "plasma_samples": adata_plasma_paired.n_obs,
        "csf_samples": adata_csf_paired.n_obs,
        "features": adata_plasma_paired.n_vars
    }
}

import json
with open(OUT_DIR / "data_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n数据统计:")
print(f"  完整数据集:")
print(f"    - 血浆: {summary['plasma_full']['patients']} 患者, {summary['plasma_full']['samples']} 样本")
print(f"    - CSF: {summary['csf_full']['patients']} 患者, {summary['csf_full']['samples']} 样本")
print(f"  配对数据集:")
print(f"    - 配对患者: {summary['paired']['patients']}")
print(f"    - 血浆样本: {summary['paired']['plasma_samples']}")
print(f"    - CSF样本: {summary['paired']['csf_samples']}")
print(f"    - 蛋白数: {summary['paired']['features']}")

print("\n" + "="*60)
print("✅ Step 0 完成！")
print("="*60)
print(f"\n输出目录: {OUT_DIR}")
print("\n下一步: 使用配对数据训练 World Model")
print("  - plasma_proteomics_paired.h5ad")
print("  - csf_proteomics_paired.h5ad")
