import os
#!/usr/bin/env python3
"""
准备治疗靶点生存分析数据

从ADNI PLASMA蛋白质组数据中提取治疗靶点表达和生存信息
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[4])

def prepare_survival_data():
    print("=" * 80)
    print("准备治疗靶点生存分析数据")
    print("=" * 80)
    
    # 1. 读取治疗靶点
    print("\n1. 读取治疗靶点...")
    therapeutic_genes_file = PROJECT_ROOT / 'output/step5_gene_classification/therapeutic_targets.csv'
    therapeutic_df = pd.read_csv(therapeutic_genes_file)
    
    # 只保留蛋白质组学治疗靶点
    proteomics_df = therapeutic_df[therapeutic_df['omics'] == 'proteomics'].copy()
    therapeutic_genes = proteomics_df['gene'].unique().tolist()
    print(f"   蛋白质组学治疗靶点: {', '.join(therapeutic_genes)}")
    
    # 2. 读取PLASMA蛋白质组数据
    print("\n2. 读取PLASMA蛋白质组数据...")
    nulisa_df = pd.read_csv(PROJECT_ROOT / 'data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv',
                            low_memory=False)
    
    # 筛选PLASMA样本和治疗靶点
    plasma_df = nulisa_df[
        (nulisa_df['SampleMatrixType'] == 'PLASMA') &
        (nulisa_df['Target'].isin(therapeutic_genes)) &
        (nulisa_df['SampleQC'] == 'passed') &
        (nulisa_df['RID'].notna())
    ].copy()
    
    print(f"   PLASMA治疗靶点数据: {len(plasma_df)} 条记录")
    
    # 3. 读取诊断数据
    print("\n3. 读取诊断数据...")
    dx_df = pd.read_csv(PROJECT_ROOT / 'data/blood-transcription-protein/DXSUM_17Apr2026.csv')
    
    # 只保留基线诊断
    dx_baseline = dx_df[dx_df['VISCODE'] == 'bl'].copy()
    
    # DIAGNOSIS编码：1=正常, 2=MCI, 3=AD
    dx_baseline['diagnosis_label'] = dx_baseline['DIAGNOSIS'].map({
        1: 'CN',
        2: 'MCI',
        3: 'AD'
    })
    
    print(f"   基线诊断: {len(dx_baseline)} 条")
    print(f"   诊断分布:")
    print(dx_baseline['diagnosis_label'].value_counts())
    
    # 4. 转换为宽格式（每个样本一行，每个基因一列）
    print("\n4. 转换数据格式...")
    
    # 只保留基线数据
    plasma_bl = plasma_df[plasma_df['VISCODE'] == 'bl'].copy()
    
    # 转换为宽格式
    plasma_wide = plasma_bl.pivot_table(
        index='RID',
        columns='Target',
        values='NPQ',
        aggfunc='first'
    ).reset_index()
    
    print(f"   宽格式数据: {len(plasma_wide)} 样本 × {len(plasma_wide.columns)-1} 基因")
    
    # 5. 合并诊断信息
    print("\n5. 合并诊断信息...")
    
    merged = plasma_wide.merge(
        dx_baseline[['RID', 'DIAGNOSIS', 'diagnosis_label']],
        on='RID',
        how='inner'
    )
    
    print(f"   成功匹配: {len(merged)} 样本")
    print(f"   诊断分布:")
    print(merged['diagnosis_label'].value_counts())
    
    # 6. 创建生存数据（简化版：使用诊断作为事件）
    print("\n6. 创建生存数据...")
    
    # 事件定义：AD=1, 其他=0
    merged['event'] = (merged['diagnosis_label'] == 'AD').astype(int)
    
    # 随访时间（简化：使用固定值，实际应从纵向数据计算）
    # 这里使用诊断状态作为代理：CN=0年, MCI=2年, AD=4年
    merged['followup_years'] = merged['diagnosis_label'].map({
        'CN': 0,
        'MCI': 2,
        'AD': 4
    })
    
    print(f"   事件数: {merged['event'].sum()} / {len(merged)}")
    print(f"   事件率: {merged['event'].mean()*100:.1f}%")
    
    # 7. 保存数据
    print("\n7. 保存数据...")
    
    output_dir = PROJECT_ROOT / 'output/step5_clinical_validation/survival_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'adni_survival_data_therapeutic_targets.csv'
    merged.to_csv(output_file, index=False)
    
    print(f"\n✅ 生存数据已保存: {output_file}")
    print(f"   样本数: {len(merged)}")
    print(f"   基因数: {len([col for col in merged.columns if col in therapeutic_genes])}")
    print(f"   可用基因: {', '.join([col for col in merged.columns if col in therapeutic_genes])}")
    
    # 8. 数据质量检查
    print("\n8. 数据质量检查...")
    for gene in therapeutic_genes:
        if gene in merged.columns:
            valid = merged[gene].notna().sum()
            mean_val = merged[gene].mean()
            std_val = merged[gene].std()
            print(f"   {gene}: {valid}/{len(merged)} 有效, 均值={mean_val:.2f}, 标准差={std_val:.2f}")
    
    # 9. 统计分析：诊断组间差异（ANOVA + t检验）
    print("\n9. 统计分析：诊断组间差异...")
    try:
        from scipy import stats
        
        stat_results = []
        
        for gene in therapeutic_genes:
            if gene not in merged.columns:
                continue
            
            # 按诊断分组
            cn_group = merged[merged['diagnosis_label']=='CN'][gene]
            mci_group = merged[merged['diagnosis_label']=='MCI'][gene]
            ad_group = merged[merged['diagnosis_label']=='AD'][gene]
            
            # ANOVA检验（三组比较）
            f_stat, p_anova = stats.f_oneway(cn_group, mci_group, ad_group)
            
            # t检验（AD vs CN）
            t_stat, p_ttest = stats.ttest_ind(ad_group, cn_group)
            
            # 效应量（Cohen's d）
            mean_diff = ad_group.mean() - cn_group.mean()
            pooled_std = np.sqrt((ad_group.std()**2 + cn_group.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            stat_results.append({
                'gene': gene,
                'CN_mean': cn_group.mean(),
                'CN_std': cn_group.std(),
                'MCI_mean': mci_group.mean(),
                'MCI_std': mci_group.std(),
                'AD_mean': ad_group.mean(),
                'AD_std': ad_group.std(),
                'mean_diff_AD_vs_CN': mean_diff,
                'cohens_d': cohens_d,
                'p_anova': p_anova,
                'p_ttest_AD_vs_CN': p_ttest
            })
            
            sig_anova = '***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else ''
            sig_ttest = '***' if p_ttest < 0.001 else '**' if p_ttest < 0.01 else '*' if p_ttest < 0.05 else ''
            print(f"   {gene}:")
            print(f"      CN={cn_group.mean():.2f}±{cn_group.std():.2f}, MCI={mci_group.mean():.2f}±{mci_group.std():.2f}, AD={ad_group.mean():.2f}±{ad_group.std():.2f}")
            print(f"      ANOVA p={p_anova:.4f} {sig_anova}, t-test(AD vs CN) p={p_ttest:.4f} {sig_ttest}, Cohen's d={cohens_d:.3f}")
        
        # 保存统计结果
        if stat_results:
            stat_df = pd.DataFrame(stat_results)
            stat_output = output_dir / 'diagnostic_group_comparison_proteomics.csv'
            stat_df.to_csv(stat_output, index=False)
            print(f"\n✅ 统计分析结果已保存: {stat_output}")
        else:
            print("\n⚠️ 无统计分析结果")
            
    except ImportError as e:
        print(f"\n⚠️ 缺少必要的库: {e}")
        print("   安装命令: pip install scipy")
    
    print("\n" + "="*80)
    print("注意：当前数据为横断面诊断数据，不适合Cox回归生存分析")
    print("已改用ANOVA和t检验分析不同诊断组间的蛋白质表达差异")
    print("="*80)
    
    return merged

if __name__ == '__main__':
    prepare_survival_data()
