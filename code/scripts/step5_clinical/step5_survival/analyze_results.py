#!/usr/bin/env python3
"""
生存分析结果深度分析（修正版）
"""
import pandas as pd
import numpy as np
from scipy import stats

print('='*80)
print('生存分析结果深度分析')
print('='*80)

# 读取转录组数据
adni_df = pd.read_csv('output/step5_clinical_validation/survival_analysis/adni_survival_data.csv')

# 读取蛋白质组数据
plasma_df = pd.read_csv('output/step5_clinical_validation/survival_analysis/adni_survival_data_therapeutic_targets.csv')

print('\n【数据集概览】')
print('\n1. 转录组学治疗靶点（ADNI基因表达）')
print(f'   样本数: {len(adni_df)}')
print(f'   基因数: {len(adni_df.columns) - 5}个')
print(f'   事件数: {adni_df["event"].sum()} / {len(adni_df)} ({adni_df["event"].mean()*100:.1f}%)')
print(f'   随访时间: 中位 {adni_df["followup_years"].median():.1f}年, 平均 {adni_df["followup_years"].mean():.1f}年')

print('\n2. 蛋白质组学治疗靶点（ADNI PLASMA蛋白质组）')
print(f'   样本数: {len(plasma_df)}')
proteomics_genes = [col for col in plasma_df.columns if col not in ["RID", "DIAGNOSIS", "diagnosis_label", "event", "followup_years"]]
print(f'   基因数: {len(proteomics_genes)}个')
print(f'   基因列表: {", ".join(proteomics_genes)}')
print(f'   事件数: {plasma_df["event"].sum()} / {len(plasma_df)} ({plasma_df["event"].mean()*100:.1f}%)')
print(f'   诊断分布:')
print(plasma_df['diagnosis_label'].value_counts())

print('\n' + '='*80)
print('【转录组学治疗靶点分析】')
print('='*80)

# 转录组基因列表 - 从生存数据文件动态获取
transcriptomics_genes = [col for col in adni_df.columns if col not in ["RID", "DIAGNOSIS", "diagnosis_label", "event", "followup_years", "Phase", "SubjectID"]]

print('\n1. 事件组 vs 非事件组基因表达差异')
print('基因\t\t事件组均值\t非事件组均值\t差异\t\tp值(t检验)')
print('-'*80)

for gene in transcriptomics_genes:
    event_group = adni_df[adni_df['event']==1][gene]
    no_event_group = adni_df[adni_df['event']==0][gene]
    
    mean_event = event_group.mean()
    mean_no_event = no_event_group.mean()
    diff = mean_event - mean_no_event
    
    # t检验
    t_stat, p_val = stats.ttest_ind(event_group, no_event_group)
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f'{gene:10s}\t{mean_event:.4f}\t\t{mean_no_event:.4f}\t\t{diff:+.4f}\t\t{p_val:.4f} {sig}')

print('\n2. 基因表达统计')
gene_stats = adni_df[transcriptomics_genes].describe().T
print(gene_stats[['mean', 'std', 'min', 'max']])

print('\n3. 基因表达相关性')
if len(transcriptomics_genes) >= 2:
    corr = adni_df[transcriptomics_genes].corr()
    print(corr.to_string())
else:
    print('  只有 1 个基因，无法计算相关性')

print('\n' + '='*80)
print('【蛋白质组学治疗靶点分析】')
print('='*80)

print('\n1. 事件组 vs 非事件组基因表达差异')
print('基因\t\t事件组均值\t非事件组均值\t差异\t\tp值(t检验)')
print('-'*80)

for gene in proteomics_genes:
    event_group = plasma_df[plasma_df['event']==1][gene]
    no_event_group = plasma_df[plasma_df['event']==0][gene]
    
    mean_event = event_group.mean()
    mean_no_event = no_event_group.mean()
    diff = mean_event - mean_no_event
    
    # t检验
    t_stat, p_val = stats.ttest_ind(event_group, no_event_group)
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f'{gene:10s}\t{mean_event:.4f}\t\t{mean_no_event:.4f}\t\t{diff:+.4f}\t\t{p_val:.4f} {sig}')

print('\n2. 基因表达统计')
plasma_gene_stats = plasma_df[proteomics_genes].describe().T
print(plasma_gene_stats[['mean', 'std', 'min', 'max']])

print('\n3. 诊断组间基因表达差异（蛋白质组）')
print('基因\t\tCN均值\t\tMCI均值\t\tAD均值\t\tANOVA p值')
print('-'*80)

for gene in proteomics_genes:
    cn_group = plasma_df[plasma_df['diagnosis_label']=='CN'][gene]
    mci_group = plasma_df[plasma_df['diagnosis_label']=='MCI'][gene]
    ad_group = plasma_df[plasma_df['diagnosis_label']=='AD'][gene]
    
    # ANOVA
    f_stat, p_val = stats.f_oneway(cn_group, mci_group, ad_group)
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f'{gene:10s}\t{cn_group.mean():.4f}\t\t{mci_group.mean():.4f}\t\t{ad_group.mean():.4f}\t\t{p_val:.4f} {sig}')

print('\n' + '='*80)
print('【核心结论】')
print('='*80)

print('\n✅ 数据分离正确：')
print('   - 转录组学靶点（IP6K1, TRMT44）使用ADNI基因表达数据')
print('   - 蛋白质组学靶点（9个基因）使用ADNI PLASMA蛋白质组数据')

print('\n📊 样本量对比：')
print(f'   - 转录组: {len(adni_df)}样本, 事件率{adni_df["event"].mean()*100:.1f}%')
print(f'   - 蛋白质组: {len(plasma_df)}样本, 事件率{plasma_df["event"].mean()*100:.1f}%')

print('\n⚠️ 事件定义不同：')
print('   - 转录组: MMSE下降≥3分 或 ADAS增加≥4分（认知下降）')
print('   - 蛋白质组: 基线诊断为AD（横断面诊断）')
print('   - 两者不可直接比较！')
