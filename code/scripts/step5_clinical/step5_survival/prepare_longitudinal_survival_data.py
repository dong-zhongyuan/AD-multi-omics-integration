import os
#!/usr/bin/env python3
"""
准备真正的纵向生存分析数据

事件定义：从CN/MCI转化为AD
随访时间：从基线到事件发生或最后随访的实际时间
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[4])

def prepare_longitudinal_survival_data():
    print("=" * 80)
    print("准备纵向生存分析数据")
    print("=" * 80)
    
    # 1. 读取治疗靶点
    print("\n1. 读取治疗靶点...")
    therapeutic_genes_file = PROJECT_ROOT / 'output/step5_gene_classification/therapeutic_targets.csv'
    therapeutic_df = pd.read_csv(therapeutic_genes_file)
    
    # 只保留蛋白质组学治疗靶点
    proteomics_df = therapeutic_df[therapeutic_df['omics'] == 'proteomics'].copy()
    therapeutic_genes = proteomics_df['gene'].unique().tolist()
    print(f"   蛋白质组学治疗靶点: {', '.join(therapeutic_genes)}")
    
    # 2. 读取诊断数据
    print("\n2. 读取诊断数据...")
    dxsum_file = PROJECT_ROOT / 'data/blood-transcription-protein/DXSUM_17Apr2026.csv'
    dxsum = pd.read_csv(dxsum_file)
    print(f"   总记录数: {len(dxsum)}")
    print(f"   受试者数: {dxsum['RID'].nunique()}")
    
    # 3. 构建生存数据：事件定义为转化为AD
    print("\n3. 构建生存数据（事件=转化为AD）...")
    
    conversion_data = []
    
    for rid in dxsum['RID'].unique():
        subject_data = dxsum[dxsum['RID'] == rid].sort_values('VISCODE2')
        
        # 获取基线诊断
        baseline = subject_data.iloc[0]
        baseline_dx = baseline['DIAGNOSIS']
        baseline_viscode = baseline['VISCODE']
        
        # 跳过基线就是AD的受试者
        if baseline_dx == 3:
            continue
        
        # 跳过只有一次访视的受试者
        if len(subject_data) < 2:
            continue
        
        # 检查是否转化为AD
        ad_visits = subject_data[subject_data['DIAGNOSIS'] == 3]
        
        if len(ad_visits) > 0:
            # 发生事件：转化为AD
            event = 1
            event_viscode = ad_visits.iloc[0]['VISCODE']
            event_date = pd.to_datetime(ad_visits.iloc[0]['EXAMDATE'])
            baseline_date = pd.to_datetime(baseline['EXAMDATE'])
            followup_years = (event_date - baseline_date).days / 365.25
        else:
            # 未发生事件：删失
            event = 0
            last_visit = subject_data.iloc[-1]
            event_viscode = last_visit['VISCODE']
            baseline_date = pd.to_datetime(baseline['EXAMDATE'])
            last_date = pd.to_datetime(last_visit['EXAMDATE'])
            followup_years = (last_date - baseline_date).days / 365.25
        
        # 只保留随访时间>0的受试者
        if followup_years <= 0:
            continue
        
        conversion_data.append({
            'RID': rid,
            'baseline_diagnosis': baseline_dx,
            'baseline_viscode': baseline_viscode,
            'event_viscode': event_viscode,
            'event': event,
            'followup_years': followup_years,
            'n_visits': len(subject_data)
        })
    
    survival_df = pd.DataFrame(conversion_data)
    
    print(f"   有效受试者数: {len(survival_df)}")
    print(f"   基线CN: {(survival_df['baseline_diagnosis']==1).sum()}")
    print(f"   基线MCI: {(survival_df['baseline_diagnosis']==2).sum()}")
    print(f"   转化为AD: {survival_df['event'].sum()}")
    print(f"   删失: {(survival_df['event']==0).sum()}")
    print(f"   转化率: {survival_df['event'].mean()*100:.1f}%")
    print(f"   平均随访: {survival_df['followup_years'].mean():.2f} 年")
    print(f"   中位随访: {survival_df['followup_years'].median():.2f} 年")
    
    # 4. 读取基线蛋白质组数据
    print("\n4. 读取基线蛋白质组数据...")
    nulisa_file = PROJECT_ROOT / 'data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv'
    nulisa_df = pd.read_csv(nulisa_file, low_memory=False)
    
    # 筛选PLASMA基线样本和治疗靶点
    plasma_baseline = nulisa_df[
        (nulisa_df['SampleMatrixType'] == 'PLASMA') &
        (nulisa_df['VISCODE'] == 'bl') &
        (nulisa_df['Target'].isin(therapeutic_genes)) &
        (nulisa_df['SampleQC'] == 'passed') &
        (nulisa_df['RID'].notna())
    ].copy()
    
    print(f"   基线PLASMA治疗靶点数据: {len(plasma_baseline)} 条记录")
    
    # 5. 转换为宽格式
    print("\n5. 转换数据格式...")
    plasma_wide = plasma_baseline.pivot_table(
        index='RID',
        columns='Target',
        values='NPQ',
        aggfunc='first'
    ).reset_index()
    
    print(f"   宽格式数据: {len(plasma_wide)} 样本 × {len(therapeutic_genes)} 基因")
    
    # 6. 合并生存数据和蛋白质组数据
    print("\n6. 合并生存数据和蛋白质组数据...")
    merged = survival_df.merge(plasma_wide, on='RID', how='inner')
    
    print(f"   成功匹配: {len(merged)} 样本")
    print(f"   基线CN: {(merged['baseline_diagnosis']==1).sum()}")
    print(f"   基线MCI: {(merged['baseline_diagnosis']==2).sum()}")
    print(f"   转化为AD: {merged['event'].sum()}")
    print(f"   转化率: {merged['event'].mean()*100:.1f}%")
    
    # 7. 保存数据
    print("\n7. 保存数据...")
    output_dir = PROJECT_ROOT / 'output/step5_clinical_validation/survival_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'longitudinal_survival_data_proteomics.csv'
    merged.to_csv(output_file, index=False)
    
    print(f"\n✅ 纵向生存数据已保存: {output_file}")
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
    
    # 9. Cox回归生存分析（多变量 + 交互作用）
    print("\n9. Cox回归生存分析（多变量 + 交互作用）...")
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import logrank_test
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        cox_results = []
        km_plots_dir = output_dir / 'kaplan_meier_plots'
        km_plots_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"全体人群 (n={len(merged)}, events={merged['event'].sum()})")
        print('='*60)
        
        for gene in therapeutic_genes:
            if gene not in merged.columns:
                continue
            
            # 准备数据
            cox_data = merged[[gene, 'followup_years', 'event', 'baseline_diagnosis']].dropna()
            
            if len(cox_data) < 10 or cox_data['event'].sum() < 5:
                print(f"   ✗ {gene}: 样本量或事件数不足 (n={len(cox_data)}, events={cox_data['event'].sum()})")
                continue
            
            # 标准化基因表达（Z-score）
            cox_data[f'{gene}_zscore'] = (cox_data[gene] - cox_data[gene].mean()) / cox_data[gene].std()
            
            # 多变量Cox回归（校正基线诊断）
            cph_multi = CoxPHFitter()
            cph_multi.fit(cox_data[[f'{gene}_zscore', 'baseline_diagnosis', 'followup_years', 'event']], 
                         duration_col='followup_years', 
                         event_col='event')
            
            summary_multi = cph_multi.summary
            hr_adj = np.exp(summary_multi.loc[f'{gene}_zscore', 'coef'])
            hr_adj_lower = np.exp(summary_multi.loc[f'{gene}_zscore', 'coef'] - 1.96 * summary_multi.loc[f'{gene}_zscore', 'se(coef)'])
            hr_adj_upper = np.exp(summary_multi.loc[f'{gene}_zscore', 'coef'] + 1.96 * summary_multi.loc[f'{gene}_zscore', 'se(coef)'])
            p_value_adjusted = summary_multi.loc[f'{gene}_zscore', 'p']
            
            # 交互作用分析（基因 × 基线诊断）
            cox_data[f'{gene}_x_dx'] = cox_data[f'{gene}_zscore'] * cox_data['baseline_diagnosis']
            
            cph_interact = CoxPHFitter()
            cph_interact.fit(cox_data[[f'{gene}_zscore', 'baseline_diagnosis', f'{gene}_x_dx', 'followup_years', 'event']], 
                            duration_col='followup_years', 
                            event_col='event')
            
            summary_interact = cph_interact.summary
            p_value_interaction = summary_interact.loc[f'{gene}_x_dx', 'p']
            hr_interaction = np.exp(summary_interact.loc[f'{gene}_x_dx', 'coef'])
            
            # 高/低表达组分析（中位数分割）
            median_val = cox_data[gene].median()
            cox_data['high_expr'] = (cox_data[gene] > median_val).astype(int)
            
            # Kaplan-Meier生存曲线
            kmf = KaplanMeierFitter()
            
            plt.figure(figsize=(10, 6))
            
            # 高表达组
            high_group = cox_data[cox_data['high_expr'] == 1]
            kmf.fit(high_group['followup_years'], high_group['event'], label=f'High {gene} (n={len(high_group)})')
            kmf.plot_survival_function(ci_show=True)
            
            # 低表达组
            low_group = cox_data[cox_data['high_expr'] == 0]
            kmf.fit(low_group['followup_years'], low_group['event'], label=f'Low {gene} (n={len(low_group)})')
            kmf.plot_survival_function(ci_show=True)
            
            # Log-rank检验
            logrank_result = logrank_test(
                high_group['followup_years'], 
                low_group['followup_years'],
                high_group['event'], 
                low_group['event']
            )
            p_value_logrank = logrank_result.p_value
            
            plt.title(f'{gene}: Kaplan-Meier Survival Curves\nLog-rank p={p_value_logrank:.4f}')
            plt.xlabel('Time to AD conversion (years)')
            plt.ylabel('Survival probability (AD-free)')
            plt.legend()
            plt.tight_layout()
            
            km_plot_file = km_plots_dir / f'{gene}_kaplan_meier.png'
            plt.savefig(km_plot_file, dpi=300)
            plt.close()
            
            cox_results.append({
                'gene': gene,
                'HR_adjusted': hr_adj,
                'HR_lower_adjusted': hr_adj_lower,
                'HR_upper_adjusted': hr_adj_upper,
                'p_value_adjusted': p_value_adjusted,
                'HR_interaction': hr_interaction,
                'p_value_interaction': p_value_interaction,
                'p_value_logrank': p_value_logrank,
                'n_samples': len(cox_data),
                'n_events': cox_data['event'].sum()
            })
            
            sig_adj = '***' if p_value_adjusted < 0.001 else '**' if p_value_adjusted < 0.01 else '*' if p_value_adjusted < 0.05 else ''
            sig_int = '***' if p_value_interaction < 0.001 else '**' if p_value_interaction < 0.01 else '*' if p_value_interaction < 0.05 else ''
            sig_log = '***' if p_value_logrank < 0.001 else '**' if p_value_logrank < 0.01 else '*' if p_value_logrank < 0.05 else ''
            
            print(f"   {gene}:")
            print(f"      多变量: HR={hr_adj:.3f} ({hr_adj_lower:.3f}-{hr_adj_upper:.3f}), p={p_value_adjusted:.4f} {sig_adj}")
            print(f"      交互作用: HR={hr_interaction:.3f}, p={p_value_interaction:.4f} {sig_int}")
            print(f"      Log-rank: p={p_value_logrank:.4f} {sig_log}")
            print(f"      KM曲线: {km_plot_file}")
        
        # 保存Cox回归结果
        if cox_results:
            cox_df = pd.DataFrame(cox_results)
            cox_output = output_dir / 'cox_multivariate_interaction_results_proteomics.csv'
            cox_df.to_csv(cox_output, index=False)
            print(f"\n✅ Cox回归结果已保存: {cox_output}")
            
            # 统计显著靶点
            print("\n" + "="*60)
            print("显著靶点汇总（p<0.05）:")
            print("="*60)
            sig_adj = cox_df[cox_df['p_value_adjusted'] < 0.05]
            sig_int = cox_df[cox_df['p_value_interaction'] < 0.05]
            sig_log = cox_df[cox_df['p_value_logrank'] < 0.05]
            
            if len(sig_adj) > 0:
                print(f"多变量显著: {', '.join(sig_adj['gene'].tolist())}")
            if len(sig_int) > 0:
                print(f"交互作用显著: {', '.join(sig_int['gene'].tolist())}")
            if len(sig_log) > 0:
                print(f"Log-rank显著: {', '.join(sig_log['gene'].tolist())}")
            if len(sig_adj) == 0 and len(sig_int) == 0 and len(sig_log) == 0:
                print("无显著靶点")
        else:
            print("\n⚠️ 无Cox回归结果")
            
    except ImportError:
        print("\n⚠️ lifelines库未安装，跳过Cox回归分析")
        print("   安装命令: pip install lifelines")
    except Exception as e:
        print(f"\n⚠️ Cox回归分析出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 纵向生存分析完成（分层分析：CN vs MCI）")
    print("事件定义：从CN/MCI转化为AD")
    print("随访时间：从基线到事件发生或最后随访的实际时间")
    print("分层：CN、MCI、All（合并）")
    print("="*80)
    
    return merged

if __name__ == '__main__':
    prepare_longitudinal_survival_data()
