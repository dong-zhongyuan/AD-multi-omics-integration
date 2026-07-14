#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5 生存分析：蛋白组 Hub 基因的 Cox 回归（基线 CN 人群）

输入：
- Step3 蛋白组 Hub 基因（脑端 + 血端，从 filtered_cross_tissue_edges.csv）
- ADNI NULISA 血浆蛋白组 + 纵向诊断

输出：
- Cox 回归结果 CSV（按 tissue_origin 分列）
- KM 曲线 PNG
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def main():
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Step5 生存分析（蛋白组 Hub 基因，基线 CN 人群）")
    print("=" * 60)

    # ================================================================
    # 1. 提取蛋白组基因，标注脑端/血端
    # ================================================================
    edges_file = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis/proteomics/filtered_cross_tissue_edges.csv"
    edges = pd.read_csv(edges_file)

    def clean_name(name):
        name = str(name).upper()
        if name.startswith("BD-"):
            name = name[3:]
        if name.startswith("PTAU") or "PTAU" in name:
            return "MAPT"
        if name.startswith("ABETA") or name.startswith("Aβ"):
            return "APP"
        return name

    brain_genes = sorted(set(edges["source"].apply(clean_name)))
    blood_genes = sorted(set(edges["target"].apply(clean_name)))
    all_genes = sorted(set(brain_genes) | set(blood_genes))

    gene_origin = {}
    for g in brain_genes:
        gene_origin[g] = 'brain'
    for g in blood_genes:
        gene_origin[g] = 'both' if g in gene_origin else 'blood'

    print(f"脑端基因: {brain_genes}")
    print(f"血端基因: {blood_genes}")
    print(f"合计: {all_genes}")

    # ================================================================
    # 2. 加载 ADNI 诊断数据 → 生存数据（事件=认知下降 MMSE↓≥3）
    # ================================================================
    print("\n[1] 构建 ADNI 生存数据（事件=MMSE下降≥3分）...")
    dxsum = pd.read_csv(PROJECT_ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")
    mmse = pd.read_csv(PROJECT_ROOT / "data/survival/MMSE_05May2026.csv")

    # 基线 MMSE
    mmse_bl = mmse[mmse['VISCODE2'] == 'bl'][['RID', 'MMSCORE']].drop_duplicates(subset='RID')
    mmse_bl.columns = ['RID', 'baseline_mmse']

    survival_rows = []
    for rid in dxsum['RID'].unique():
        sub = dxsum[dxsum['RID'] == rid].sort_values('VISCODE2')
        baseline = sub.iloc[0]
        baseline_date = pd.to_datetime(baseline['EXAMDATE'])

        # 基线 MMSE
        bl_mmse_row = mmse_bl[mmse_bl['RID'] == rid]
        if bl_mmse_row.empty:
            continue
        baseline_mmse = bl_mmse_row.iloc[0]['baseline_mmse']

        # 跟踪 MMSE 下降
        mmse_sub = mmse[mmse['RID'] == rid].sort_values('VISCODE2')
        event = 0
        event_date = None
        for _, row in mmse_sub.iterrows():
            if row['VISCODE2'] == 'bl':
                continue
            if pd.isna(row['MMSCORE']):
                continue
            drop = baseline_mmse - row['MMSCORE']
            if drop >= 3:
                event = 1
                # 找对应日期
                visit_code = row['VISCODE2']
                visit_date_row = sub[sub['VISCODE2'] == visit_code]
                if not visit_date_row.empty:
                    event_date = pd.to_datetime(visit_date_row.iloc[0]['EXAMDATE'])
                else:
                    # 估算：从 VISCODE 推断月份
                    event_date = baseline_date
                break

        if event_date is not None:
            followup = (event_date - baseline_date).days / 365.25
        else:
            last = sub.iloc[-1]
            followup = (pd.to_datetime(last['EXAMDATE']) - baseline_date).days / 365.25

        if followup > 0:
            survival_rows.append({'RID': rid, 'event': event, 'followup_years': followup})

    survival_df = pd.DataFrame(survival_rows)
    print(f"  基线 CN: {len(survival_df)}, events: {survival_df['event'].sum()}")

    # ================================================================
    # 3. 加载 NULISA 血浆蛋白组
    # ================================================================
    print("\n[2] 加载血浆蛋白组...")
    nulisa = pd.read_csv(
        PROJECT_ROOT / "data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv",
        low_memory=False
    )
    plasma_bl = nulisa[
        (nulisa['SampleMatrixType'] == 'PLASMA') &
        (nulisa['VISCODE'] == 'bl') &
        (nulisa['SampleQC'] == 'passed') &
        (nulisa['RID'].notna()) &
        (nulisa['Target'].isin(all_genes))
    ].copy()
    plasma_bl['RID'] = plasma_bl['RID'].astype(int)

    plasma_wide = plasma_bl.pivot_table(index='RID', columns='Target', values='NPQ', aggfunc='first').reset_index()
    print(f"  血浆蛋白组: {len(plasma_wide)} 样本 × {len([c for c in plasma_wide.columns if c != 'RID'])} 蛋白")

    # ================================================================
    # 4. 合并 + 加 APOE4 校正
    # ================================================================
    merged = survival_df.merge(plasma_wide, on='RID', how='inner')

    # 加载 APOE4 携带状态
    apoe_df = pd.read_csv(PROJECT_ROOT / 'data/survival/APOERES_05May2026.csv')
    apoe_df['APOE4_carrier'] = apoe_df['GENOTYPE'].astype(str).apply(lambda x: 1 if '4' in x else 0)
    apoe_summary = apoe_df[['RID', 'APOE4_carrier']].drop_duplicates(subset='RID')
    merged = merged.merge(apoe_summary, on='RID', how='left')
    merged['APOE4_carrier'] = merged['APOE4_carrier'].fillna(0).astype(int)

    available_genes = [g for g in all_genes if g in merged.columns]
    print(f"\n[3] 合并后: {len(merged)} 样本, {len(available_genes)} 可用基因")
    print(f"  APOE4 携带者: {merged['APOE4_carrier'].sum()}/{len(merged)}")

    # ================================================================
    # 5. Cox 回归（分脑端/血端报告）
    # ================================================================
    output_dir = PROJECT_ROOT / "output/step5_clinical_validation/survival_analysis"
    km_dir = output_dir / "kaplan_meier_plots"
    km_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 只做血端基因生存分析
    blood_only = [g for g in available_genes if gene_origin.get(g) in ('blood', 'both')]
    for group_name, group_genes in [
        ('Blood-end', blood_only)
    ]:
        print(f"\n{'='*60}")
        print(f"{group_name} (n={len(merged)}, events={merged['event'].sum()})")
        print(f"基因: {group_genes}")
        print('='*60)

        for gene in group_genes:
            data = merged[[gene, 'followup_years', 'event', 'APOE4_carrier']].dropna()
            if len(data) < 10 or data['event'].sum() < 3:
                print(f"  {gene}: 样本或事件不足，跳过")
                continue

            # Z-score
            data[f'{gene}_z'] = (data[gene] - data[gene].mean()) / data[gene].std()

            # Cox（校正 APOE4）
            cph = CoxPHFitter()
            cph.fit(data[[f'{gene}_z', 'APOE4_carrier', 'followup_years', 'event']],
                    duration_col='followup_years', event_col='event')
            s = cph.summary.loc[f'{gene}_z']
            hr = np.exp(s['coef'])
            hr_lo = np.exp(s['coef'] - 1.96 * s['se(coef)'])
            hr_hi = np.exp(s['coef'] + 1.96 * s['se(coef)'])
            p_cox = s['p']

            # Log-rank (中位数分组)
            median = data[gene].median()
            high = data[data[gene] >= median]
            low = data[data[gene] < median]
            if len(high) > 0 and len(low) > 0 and high['event'].sum() > 0 and low['event'].sum() > 0:
                lr = logrank_test(high['followup_years'], low['followup_years'],
                                  high['event'], low['event'])
                p_lr = lr.p_value
            else:
                p_lr = 1.0

            sig_cox = '***' if p_cox < 0.001 else '**' if p_cox < 0.01 else '*' if p_cox < 0.05 else ''
            sig_lr = '***' if p_lr < 0.001 else '**' if p_lr < 0.01 else '*' if p_lr < 0.05 else ''

            print(f"  {gene} [{gene_origin.get(gene,'?')}]: HR={hr:.3f} ({hr_lo:.3f}-{hr_hi:.3f}), p_cox={p_cox:.4f} {sig_cox}, p_logrank={p_lr:.4f} {sig_lr}")

            results.append({
                'gene': gene,
                'tissue_origin': gene_origin.get(gene, 'unknown'),
                'HR': hr,
                'HR_lower': hr_lo,
                'HR_upper': hr_hi,
                'p_cox': p_cox,
                'p_logrank': p_lr,
                'n': len(data),
                'events': int(data['event'].sum()),
            })

            # KM 曲线
            fig, ax = plt.subplots(figsize=(5, 4))
            for label, grp in [('High', high), ('Low', low)]:
                kmf = KaplanMeierFitter()
                kmf.fit(grp['followup_years'], grp['event'], label=label)
                kmf.plot_survival_function(ax=ax)
            ax.set_title(f'{gene} ({gene_origin.get(gene,"?")})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Years')
            ax.set_ylabel('Survival probability')
            plt.tight_layout()
            safe_gene = gene.replace('?', '').replace('Î²', 'beta').replace('β', 'beta').replace('/', '_')
            plt.savefig(km_dir / f'{safe_gene}_kaplan_meier.png', dpi=300, bbox_inches='tight')
            plt.close()

    # ================================================================
    # 6. 保存 + 汇总
    # ================================================================
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'cox_results_proteomics.csv', index=False)
    print(f"\n✅ 保存: {output_dir / 'cox_results_proteomics.csv'}")

    print("\n" + "=" * 60)
    print("汇总（基线 CN 人群）")
    print("=" * 60)
    print(results_df[['gene', 'tissue_origin', 'HR', 'p_cox', 'p_logrank']].to_string(index=False))

    sig = results_df[(results_df['p_cox'] < 0.05) | (results_df['p_logrank'] < 0.05)]
    if len(sig) > 0:
        print(f"\n显著基因 (p_cox<0.05 或 p_logrank<0.05):")
        print(sig[['gene', 'tissue_origin', 'HR', 'p_cox', 'p_logrank']].to_string(index=False))
    else:
        print("\n无显著基因")


if __name__ == '__main__':
    main()
