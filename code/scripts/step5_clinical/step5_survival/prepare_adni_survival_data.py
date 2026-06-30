#!/usr/bin/env python3
# ============================================================================
# ADNI生存分析数据准备 (Python版本 - 修复版)
# ============================================================================

import sys
sys.path.insert(0, "")
from tools.config_loader import get_config
config = get_config()

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 设置路径
data_dir = str(config.get_path("paths.data_dir")) + "/survival"
output_dir = str(config.get_path("paths.output_dir")) + "/step5_clinical_validation/survival_analysis"
os.makedirs(output_dir, exist_ok=True)

print("=== ADNI生存分析数据准备 ===\n")

# ============================================================================
# 1. 读取基因表达数据
# ============================================================================
print("1. 读取基因表达数据...")

expr_file = os.path.join(data_dir, "ADNI_Gene_Expression_Profile.csv")
print(f"   读取文件: {expr_file}")
print(f"   文件大小: {os.path.getsize(expr_file) / 1024**2:.1f} MB")

# 读取完整数据（不指定header，因为格式特殊）
expr_data = pd.read_csv(expr_file, header=None, low_memory=False)
print(f"   数据维度: {expr_data.shape[0]} 行 × {expr_data.shape[1]} 列")

# 提取样本信息
# 第0行: Phase, 第1行: Visit, 第2行: SubjectID
sample_ids = expr_data.iloc[2, 3:].astype(str).values
visit_codes = expr_data.iloc[1, 3:].astype(str).values
phases = expr_data.iloc[0, 3:].astype(str).values

print(f"   样本数: {len(sample_ids)}")
print(f"   前5个样本: {', '.join(sample_ids[:5])}")

# 提取表达数据（第9行开始，索引从8开始）
# 第8行（索引8）是标题行（ProbeSet, LocusLink, Symbol）
# 第10行开始（索引9）是实际probe数据
expr_matrix = expr_data.iloc[8:, :].copy()
# 第一行作为列名
expr_matrix.columns = expr_matrix.iloc[0, :3].tolist() + list(sample_ids)
# 删除标题行，保留数据行
expr_matrix = expr_matrix.iloc[1:, :].copy()

print(f"   基因/探针数: {expr_matrix.shape[0]}")

# ============================================================================
# 2. 提取候选基因表达数据
# ============================================================================
print("\n2. 提取候选基因表达数据...")
# 从治疗靶点文件动态读取（所有omics）
tier_file = str(config.get_path("paths.output_dir")) + "/step5_gene_classification/therapeutic_targets.csv"
tier_df = pd.read_csv(tier_file)

# 取所有治疗靶点基因（不限omics）
candidate_genes_list = tier_df['gene'].unique().tolist()
# CAVIN2旧名SDPR，需要额外映射
gene_alias = {'CAVIN2': 'SDPR'}
print(f"  从 {tier_file} 读取候选基因")
print(f"  总计: {len(candidate_genes_list)} 个基因")
print(f"  基因列表: {', '.join(candidate_genes_list)}")

# 动态从ADNI数据中查找每个基因的probe ID
print("\n  动态查找probe映射...")
candidate_genes = {}
for gene in candidate_genes_list:
    # 先用原名查找
    gene_probes = expr_matrix[expr_matrix["Symbol"] == gene]["ProbeSet"].tolist()
    # 如果没找到，尝试别名
    if not gene_probes and gene in gene_alias:
        alias = gene_alias[gene]
        gene_probes = expr_matrix[expr_matrix["Symbol"] == alias]["ProbeSet"].tolist()
        if gene_probes:
            print(f"  ✓ {gene} (alias={alias}): {len(gene_probes)} 个probe")
    if gene_probes:
        candidate_genes[gene] = gene_probes
        if gene not in gene_alias or not gene_probes:
            print(f"  ✓ {gene}: {len(gene_probes)} 个probe")
    else:
        print(f"  ✗ {gene}: 未找到probe")

# 只保留有probe映射的基因到gene_list
gene_list = list(candidate_genes.keys())

print(f"\n  成功映射: {len(candidate_genes)}/{len(candidate_genes_list)} 个基因")

gene_expr_dict = {}

for gene, probes in candidate_genes.items():
    print(f"   {gene}: {len(probes)} 个探针")
    
    # 提取探针数据
    gene_data = expr_matrix[expr_matrix["ProbeSet"].isin(probes)].copy()
    
    if len(gene_data) == 0:
        print(f"      警告: 未找到探针数据")
        continue
    
    # 转换为数值矩阵（跳过前3列注释）
    expr_values = gene_data.iloc[:, 3:].copy()
    # 逐列转换为数值
    for col in expr_values.columns:
        expr_values[col] = pd.to_numeric(expr_values[col], errors='coerce')
    
    # 如果有多个探针，取平均值
    if len(gene_data) > 1:
        gene_expr = expr_values.mean(axis=0, skipna=True).values
        print(f"      使用 {len(gene_data)} 个探针的平均值")
    else:
        gene_expr = expr_values.iloc[0, :].values
        print(f"      使用单个探针")
    
    gene_expr_dict[gene] = gene_expr

# 构建基因表达数据框（动态）
gene_expr_df_dict = {
    "SubjectID": sample_ids,
    "Visit": visit_codes,
    "Phase": phases
}
# 添加所有有probe映射的基因
for gene in gene_list:
    gene_expr_df_dict[gene] = gene_expr_dict.get(gene, np.nan)

gene_expr_df = pd.DataFrame(gene_expr_df_dict)

print(f"   基因表达数据框: {gene_expr_df.shape[0]} 行 × {gene_expr_df.shape[1]} 列")
print(f"   前5行:\n{gene_expr_df.head()}")

# 保存基因表达数据
gene_expr_df.to_csv(os.path.join(output_dir, "adni_gene_expression.csv"), index=False)
print(f"   已保存: {os.path.join(output_dir, 'adni_gene_expression.csv')}")

# ============================================================================
# 3. 读取临床数据
# ============================================================================
print("\n3. 读取临床数据...")

registry = pd.read_csv(os.path.join(data_dir, "REGISTRY_05May2026.csv"))
print(f"   REGISTRY: {registry.shape[0]} 行 × {registry.shape[1]} 列")

npstatus = pd.read_csv(os.path.join(data_dir, "NPSTATUS_05May2026.csv"))
print(f"   NPSTATUS: {npstatus.shape[0]} 行 × {npstatus.shape[1]} 列")

mmse = pd.read_csv(os.path.join(data_dir, "MMSE_05May2026.csv"))
print(f"   MMSE: {mmse.shape[0]} 行 × {mmse.shape[1]} 列")

adas = pd.read_csv(os.path.join(data_dir, "ADAS_05May2026.csv"))
print(f"   ADAS: {adas.shape[0]} 行 × {adas.shape[1]} 列")

# ============================================================================
# 4. 构建生存分析数据集
# ============================================================================
print("\n4. 构建生存分析数据集...")

# 从SubjectID中提取RID
gene_expr_df["RID"] = gene_expr_df["SubjectID"].str.extract(r"_S_(\d+)", expand=False)
gene_expr_df["RID"] = pd.to_numeric(gene_expr_df["RID"], errors='coerce').astype('Int64')
# 删除RID为NaN的行
gene_expr_df = gene_expr_df[gene_expr_df["RID"].notna()].copy()
print(f"   提取RID: {gene_expr_df['RID'].nunique()} 个唯一受试者")

# 合并REGISTRY数据（获取基线信息）
registry_baseline = registry[registry["VISCODE"].isin(["bl", "sc"])].copy()
registry_baseline = registry_baseline.groupby("RID").first().reset_index()
registry_baseline = registry_baseline[["RID", "PTID", "PHASE", "VISCODE", "EXAMDATE", "RGSTATUS"]]

print(f"   基线访视: {len(registry_baseline)} 个受试者")

# 合并基因表达数据和基线信息
survival_data = gene_expr_df.merge(registry_baseline, on="RID", how="left")
print(f"   合并后: {len(survival_data)} 行")

# 合并NPSTATUS（神经病理状态）
npstatus_summary = npstatus.groupby("RID").agg({
    "NPDECIDE": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan,
    "NPDEC": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
}).reset_index()

survival_data = survival_data.merge(npstatus_summary, on="RID", how="left")
print(f"   合并NPSTATUS后: {len(survival_data)} 行")

# 合并MMSE基线评分
mmse_baseline = mmse[mmse["VISCODE"].isin(["bl", "sc"])].copy()
mmse_baseline = mmse_baseline.groupby("RID").agg({
    "MMSCORE": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
}).reset_index()
mmse_baseline.rename(columns={"MMSCORE": "MMSE_baseline"}, inplace=True)

survival_data = survival_data.merge(mmse_baseline, on="RID", how="left")

# 合并ADAS基线评分
adas_baseline = adas[adas["VISCODE"].isin(["bl", "sc"])].copy()
adas_baseline = adas_baseline.groupby("RID").agg({
    "TOTAL13": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
}).reset_index()
adas_baseline.rename(columns={"TOTAL13": "ADAS_baseline"}, inplace=True)

survival_data = survival_data.merge(adas_baseline, on="RID", how="left")

print(f"   合并认知评分后: {len(survival_data)} 行")

# ============================================================================
# 5. 定义生存时间和事件
# ============================================================================
print("\n5. 定义生存时间和事件...")

# 计算每个受试者的随访时间
registry["EXAMDATE"] = pd.to_datetime(registry["EXAMDATE"], errors="coerce")
registry_followup = registry.groupby("RID").agg({
    "EXAMDATE": ["min", "max", "count"]
}).reset_index()
registry_followup.columns = ["RID", "baseline_date", "last_visit_date", "n_visits"]
registry_followup["followup_days"] = (registry_followup["last_visit_date"] - registry_followup["baseline_date"]).dt.days
registry_followup["followup_years"] = registry_followup["followup_days"] / 365.25

print(f"   随访时间统计:")
print(f"      中位随访时间: {registry_followup['followup_years'].median():.1f} 年")
print(f"      最长随访时间: {registry_followup['followup_years'].max():.1f} 年")

survival_data = survival_data.merge(registry_followup[["RID", "followup_years", "n_visits"]], on="RID", how="left")

# 定义事件：认知下降或转化为痴呆
mmse_change = mmse.groupby("RID").agg({
    "MMSCORE": ["first", "last"]
}).reset_index()
mmse_change.columns = ["RID", "MMSE_first", "MMSE_last"]
mmse_change["MMSE_change"] = mmse_change["MMSE_last"] - mmse_change["MMSE_first"]

adas_change = adas.groupby("RID").agg({
    "TOTAL13": ["first", "last"]
}).reset_index()
adas_change.columns = ["RID", "ADAS_first", "ADAS_last"]
adas_change["ADAS_change"] = adas_change["ADAS_last"] - adas_change["ADAS_first"]

survival_data = survival_data.merge(mmse_change[["RID", "MMSE_change"]], on="RID", how="left")
survival_data = survival_data.merge(adas_change[["RID", "ADAS_change"]], on="RID", how="left")

# 定义事件：
# 1. MMSE下降≥3分（临床显著下降）
# 2. ADAS增加≥4分（临床显著恶化）
survival_data["event_mmse"] = ((survival_data["MMSE_change"] <= -3) & (survival_data["MMSE_change"].notna())).astype(int)
survival_data["event_adas"] = ((survival_data["ADAS_change"] >= 4) & (survival_data["ADAS_change"].notna())).astype(int)
survival_data["event"] = ((survival_data["event_mmse"] == 1) | (survival_data["event_adas"] == 1)).astype(int)

print(f"   事件定义:")
print(f"      MMSE下降≥3分: {survival_data['event_mmse'].sum()} 例")
print(f"      ADAS增加≥4分: {survival_data['event_adas'].sum()} 例")
print(f"      总事件数: {survival_data['event'].sum()} 例")

# ============================================================================
# 6. 数据清理和最终数据集
# ============================================================================
print("\n6. 数据清理和最终数据集...")

# 只保留有完整数据的样本
# 动态构建过滤条件（gene_list已在前面定义）
# 先检查每个基因有多少非NaN值
print(f"   基因表达数据可用性:")
genes_with_data = []
for gene in gene_list:
    n_valid = survival_data[gene].notna().sum()
    print(f"     {gene}: {n_valid}/{len(survival_data)} ({n_valid/len(survival_data)*100:.1f}%)")
    if n_valid > 0:
        genes_with_data.append(gene)

print(f"   使用有数据的基因: {len(genes_with_data)}/{len(gene_list)}")

# 只对有数据的基因进行过滤
filter_conditions = [survival_data[gene].notna() for gene in genes_with_data]
filter_conditions.append(survival_data["followup_years"].notna())
filter_conditions.append(survival_data["followup_years"] > 0)

# 合并所有条件
combined_filter = filter_conditions[0]
for condition in filter_conditions[1:]:
    combined_filter = combined_filter & condition

survival_final = survival_data[combined_filter].copy()

# 更新gene_list为实际有数据的基因
gene_list = genes_with_data

# 动态构建列列表
columns_to_keep = ["RID", "SubjectID", "Visit", "Phase"] + gene_list + [
    "MMSE_baseline", "ADAS_baseline",
    "followup_years", "event",
    "MMSE_change", "ADAS_change",
    "event_mmse", "event_adas",
    "n_visits"
]

survival_final = survival_final[columns_to_keep]

print(f"   最终数据集: {len(survival_final)} 个样本")
print(f"   事件数: {survival_final['event'].sum()} 例")
print(f"   事件率: {survival_final['event'].mean() * 100:.1f}%")

# 保存最终数据集
survival_final.to_csv(os.path.join(output_dir, "adni_survival_data.csv"), index=False)
print(f"   已保存: {os.path.join(output_dir, 'adni_survival_data.csv')}")

# ============================================================================
# 7. 描述性统计
# ============================================================================
print("\n7. 描述性统计...\n")

# 基因表达统计（只有当有基因数据时才计算）
if len(gene_list) > 0:
    gene_stats = survival_final[gene_list].describe().T
    print("基因表达统计:")
    print(gene_stats)
    # 保存统计结果
    gene_stats.to_csv(os.path.join(output_dir, "gene_expression_stats.csv"))
else:
    print("警告: 没有基因表达数据，跳过基因统计")

# 临床特征统计
clinical_stats = pd.DataFrame({
    "n": [len(survival_final)],
    "followup_mean": [survival_final["followup_years"].mean()],
    "followup_sd": [survival_final["followup_years"].std()],
    "followup_median": [survival_final["followup_years"].median()],
    "mmse_mean": [survival_final["MMSE_baseline"].mean()],
    "mmse_sd": [survival_final["MMSE_baseline"].std()],
    "adas_mean": [survival_final["ADAS_baseline"].mean()],
    "adas_sd": [survival_final["ADAS_baseline"].std()],
    "event_n": [survival_final["event"].sum()],
    "event_rate": [survival_final["event"].mean() * 100]
})

print("\n临床特征统计:")
print(clinical_stats)

# 保存统计结果
clinical_stats.to_csv(os.path.join(output_dir, "clinical_stats.csv"), index=False)

# 保存基因tier信息（只有当有基因数据时）
if len(gene_list) > 0:
    gene_tier_info = pd.DataFrame({
        'gene': gene_list,
        'tier': ['Therapeutic'] * len(gene_list)
    })
    gene_tier_info.to_csv(os.path.join(output_dir, "gene_tier_info.csv"), index=False)
    print(f"   已保存: {os.path.join(output_dir, 'gene_tier_info.csv')}")

print("\n=== 数据准备完成 ===")
print("输出文件:")
print("  1. adni_gene_expression.csv - 基因表达数据")
print("  2. adni_survival_data.csv - 生存分析数据集")
if len(gene_list) > 0:
    print("  3. gene_expression_stats.csv - 基因表达统计")
print("  4. clinical_stats.csv - 临床特征统计")
if len(gene_list) > 0:
    print("  5. gene_tier_info.csv - 基因tier信息")
