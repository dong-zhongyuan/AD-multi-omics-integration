#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5 生存分析：转录组 Hub 基因的 Cox 回归（基线 CN 人群, 血端基因）
优化版：只加载需要的基因 + 向量化生存数据 + 无画图 + 进度条

输入：
- Step3 转录组 filtered_cross_tissue_edges.csv（血端基因）
- ADNI 基因表达微阵列 + 纵向诊断 + MMSE + APOE4

输出：
- cox_results_transcriptomics.csv
- transcriptomics_survival_sig_genes.csv（p<0.05 的基因列表）
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_survival_data_vectorized(dxsum, mmse):
    """向量化构建生存数据（事件=MMSE下降≥3分）。
    
    旧版逐行 iterrows 遍历每个受试者→极慢。
    新版用 groupby + 向量化操作。
    """
    print("  [1a] 预处理 DXSUM...")
    dx = dxsum[['RID', 'VISCODE2', 'EXAMDATE']].copy()
    dx['EXAMDATE'] = pd.to_datetime(dx['EXAMDATE'], errors='coerce')
    
    print("  [1b] 预处理 MMSE...")
    ms = mmse[['RID', 'VISCODE2', 'MMSCORE']].copy()
    ms['MMSCORE'] = pd.to_numeric(ms['MMSCORE'], errors='coerce')
    
    # 基线 MMSE (ADNI screening visit = 'sc'; 有些队列用 'bl')
    ms_bl = ms[ms['VISCODE2'].isin(['sc', 'bl'])][['RID', 'MMSCORE']].drop_duplicates(subset='RID')
    ms_bl.columns = ['RID', 'baseline_mmse']
    ms_bl = ms_bl.set_index('RID')['baseline_mmse']
    
    print("  [1c] 向量化构建生存数据...")
    all_rids = dx['RID'].unique()
    survival_rows = []
    
    for rid in tqdm(all_rids, desc="  受试者", file=sys.stdout):
        if rid not in ms_bl.index:
            continue
        baseline_mmse = ms_bl[rid]
        if pd.isna(baseline_mmse):
            continue

        sub_dx = dx[dx['RID'] == rid].sort_values('VISCODE2')
        if sub_dx.empty:
            continue
        baseline_date = sub_dx.iloc[0]['EXAMDATE']
        if pd.isna(baseline_date):
            continue

        # MMSE 随访（排除基线 sc/bl）
        ms_sub = ms[(ms['RID'] == rid) & (~ms['VISCODE2'].isin(['sc', 'bl']))].sort_values('VISCODE2')

        event = 0
        event_date = None
        for _, row in ms_sub.iterrows():
            if pd.isna(row['MMSCORE']):
                continue
            drop = baseline_mmse - row['MMSCORE']
            if drop >= 3:
                event = 1
                visit_row = sub_dx[sub_dx['VISCODE2'] == row['VISCODE2']]
                event_date = visit_row.iloc[0]['EXAMDATE'] if not visit_row.empty else baseline_date
                break

        if event_date is not None:
            followup = (event_date - baseline_date).days / 365.25
        else:
            last = sub_dx.iloc[-1]
            followup = (last['EXAMDATE'] - baseline_date).days / 365.25

        if followup > 0:
            survival_rows.append({'RID': rid, 'event': event, 'followup_years': followup})

    return pd.DataFrame(survival_rows)


def load_expression_fast(gene_list):
    """高效加载 ADNI 微阵列——只读取需要的基因行。
    
    旧版加载整个 212MB 文件再 groupby，卡死在 to_numeric。
    新版：逐行扫描，只保留目标基因 + 向量化转换。
    """
    print("  [2a] 扫描 ADNI 微阵列（只保留目标基因）...")
    expr_path = PROJECT_ROOT / 'data/survival/ADNI_Gene_Expression_Profile.csv'

    # 第一遍：读 header 获取 SubjectID
    header_df = pd.read_csv(expr_path, nrows=3, low_memory=False)
    sid_row = header_df[header_df.iloc[:, 0] == 'SubjectID']
    if sid_row.empty:
        # 可能第一行就是 SubjectID
        sid_row = header_df[header_df.iloc[:, 0].astype(str).str.startswith('Subject')]
    subject_ids = sid_row.iloc[0, 3:].tolist()
    print(f"  受试者列数: {len(subject_ids)}")

    # 转为 set 加速查找
    gene_set = set(gene_list)

    # 逐 chunk 读取，只保留目标基因行
    print("  [2b] 逐块过滤探针行...")
    kept_chunks = []
    chunk_iter = pd.read_csv(expr_path, skiprows=[1], chunksize=5000, low_memory=False)
    
    for chunk in tqdm(chunk_iter, desc="  探针块", file=sys.stdout):
        # 只保留 _at 结尾的探针行
        probe_mask = chunk.iloc[:, 0].astype(str).str.endswith('_at')
        if not probe_mask.any():
            continue
        probes = chunk[probe_mask].copy()
        # 检查 Symbol 列（第3列）是否在目标基因里
        symbol_col = probes.columns[2]
        in_target = probes[symbol_col].isin(gene_set)
        if not in_target.any():
            continue
        kept_chunks.append(probes[in_target])

    if not kept_chunks:
        raise ValueError("ADNI 微阵列中未找到任何目标基因")

    raw_filtered = pd.concat(kept_chunks, ignore_index=True)
    print(f"  过滤后探针行: {len(raw_filtered)}")

    # 重命名列
    raw_filtered.columns = ['ProbeSet', 'LocusLink', 'Symbol'] + list(raw_filtered.columns[3:])
    expr = raw_filtered[['Symbol'] + list(raw_filtered.columns[3:])].copy()

    # 向量化数值转换（只对需要的行做）
    print("  [2c] 数值化表达矩阵...")
    expr_numeric = expr.copy()
    for col in tqdm(list(raw_filtered.columns[3:]), desc="  受试者列", file=sys.stdout):
        expr_numeric[col] = pd.to_numeric(expr[col], errors='coerce')

    # 按 Symbol 取均值（同一基因多个探针）
    expr_numeric = expr_numeric.groupby('Symbol').mean()
    print(f"  最终: {expr_numeric.shape[0]} 基因 × {expr_numeric.shape[1]} 受试者")

    return expr_numeric, subject_ids


def main():
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    t0 = time.time()

    print("=" * 60)
    print("Step5 生存分析（转录组血端基因, ADNI 表达队列）")
    print("=" * 60)

    # ================================================================
    # 1. 提取转录组血端基因
    # ================================================================
    print("\n[0] 提取转录组靶点基因...")
    edges_file = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis/transcriptomics/filtered_cross_tissue_edges.csv"
    edges = pd.read_csv(edges_file)

    brain_genes = sorted(set(edges["source"].astype(str)))
    blood_genes = sorted(set(edges["target"].astype(str)))
    all_genes = sorted(set(brain_genes) | set(blood_genes))

    gene_origin = {}
    for g in brain_genes:
        gene_origin[g] = 'brain'
    for g in blood_genes:
        gene_origin[g] = 'both' if g in gene_origin else 'blood'

    # 只做血端基因（用户要求）
    blood_only = [g for g in all_genes if gene_origin.get(g) in ('blood', 'both')]
    print(f"  脑端: {len(brain_genes)}, 血端: {len(blood_only)} (含 both)")
    print(f"  血端基因（前30）: {blood_only[:30]}...")

    # ================================================================
    # 2. 构建 ADNI 生存数据
    # ================================================================
    print("\n[1] 构建 ADNI 生存数据（事件=MMSE下降≥3分）...")
    dxsum = pd.read_csv(PROJECT_ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")
    mmse = pd.read_csv(PROJECT_ROOT / "data/survival/MMSE_05May2026.csv")
    survival_df = build_survival_data_vectorized(dxsum, mmse)
    print(f"  生存数据: {len(survival_df)} 受试者, events: {survival_df['event'].sum()}")
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # ================================================================
    # 3. 高效加载表达矩阵（只读目标基因）
    # ================================================================
    print(f"\n[2] 加载 ADNI 基因表达微阵列（只加载 {len(blood_only)} 个目标基因）...")
    expr_numeric, subject_ids = load_expression_fast(blood_only)

    # SubjectID → RID
    registry = pd.read_csv(PROJECT_ROOT / "data/survival/REGISTRY_05May2026.csv")
    sid_to_rid = registry[['PTID', 'RID']].drop_duplicates(subset='PTID').set_index('PTID')['RID'].to_dict()
    rid_list = [sid_to_rid.get(sid) for sid in subject_ids]
    expr_numeric.columns = rid_list
    expr_numeric = expr_numeric.loc[:, pd.notna(expr_numeric.columns)]
    expr_numeric.columns = [int(c) for c in expr_numeric.columns]
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # 转置为 RID × gene
    expr_t = expr_numeric.T.reset_index()
    expr_t.columns = ['RID'] + list(expr_t.columns[1:])
    expr_t['RID'] = expr_t['RID'].astype(int)

    # ================================================================
    # 4. 合并 + APOE4 校正
    # ================================================================
    print("\n[3] 合并生存 + 表达 + APOE4...")
    merged = survival_df.merge(expr_t, on='RID', how='inner')

    apoe_df = pd.read_csv(PROJECT_ROOT / 'data/survival/APOERES_05May2026.csv')
    apoe_df['APOE4_carrier'] = apoe_df['GENOTYPE'].astype(str).apply(lambda x: 1 if '4' in x else 0)
    apoe_summary = apoe_df[['RID', 'APOE4_carrier']].drop_duplicates(subset='RID')
    merged = merged.merge(apoe_summary, on='RID', how='left')
    merged['APOE4_carrier'] = merged['APOE4_carrier'].fillna(0).astype(int)

    available_genes = [g for g in blood_only if g in merged.columns]
    print(f"  合并后: {len(merged)} 样本, {len(available_genes)} 可用血端基因")
    print(f"  ⏱ 已用时: {time.time()-t0:.1f}s")

    # ================================================================
    # 5. Cox 回归（逐基因，带进度条）
    # ================================================================
    print(f"\n[4] Cox 回归（APOE4 校正, {len(available_genes)} 基因）...")

    output_dir = PROJECT_ROOT / "output/step5_clinical_validation/survival_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0

    for gene in tqdm(available_genes, desc="  Cox回归", file=sys.stdout):
        data = merged[[gene, 'followup_years', 'event', 'APOE4_carrier']].dropna()
        if len(data) < 10 or data['event'].sum() < 3:
            skipped += 1
            continue

        # Z-score
        data[f'{gene}_z'] = (data[gene] - data[gene].mean()) / data[gene].std()

        try:
            # Cox（校正 APOE4）
            cph = CoxPHFitter()
            cph.fit(data[[f'{gene}_z', 'APOE4_carrier', 'followup_years', 'event']],
                    duration_col='followup_years', event_col='event')
            s = cph.summary.loc[f'{gene}_z']
            hr = float(np.exp(s['coef']))
            hr_lo = float(np.exp(s['coef'] - 1.96 * s['se(coef)']))
            hr_hi = float(np.exp(s['coef'] + 1.96 * s['se(coef)']))
            p_cox = float(s['p'])

            # Log-rank (中位数分组)
            median = data[gene].median()
            high = data[data[gene] >= median]
            low = data[data[gene] < median]
            if len(high) > 0 and len(low) > 0 and high['event'].sum() > 0 and low['event'].sum() > 0:
                lr = logrank_test(high['followup_years'], low['followup_years'],
                                  high['event'], low['event'])
                p_lr = float(lr.p_value)
            else:
                p_lr = 1.0

            results.append({
                'gene': gene,
                'tissue_origin': gene_origin.get(gene, 'blood'),
                'HR': hr,
                'HR_lower': hr_lo,
                'HR_upper': hr_hi,
                'p_cox': p_cox,
                'p_logrank': p_lr,
                'n': len(data),
                'events': int(data['event'].sum()),
            })
        except Exception as e:
            skipped += 1

    print(f"\n  完成: {len(results)} 基因, 跳过: {skipped}")
    print(f"  ⏱ 总用时: {time.time()-t0:.1f}s")

    # ================================================================
    # 6. 保存结果
    # ================================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_cox')
    results_df.to_csv(output_dir / 'cox_results_transcriptomics.csv', index=False)
    print(f"\n✅ 保存: {output_dir / 'cox_results_transcriptomics.csv'}")

    # 显著基因
    sig = results_df[(results_df['p_cox'] < 0.05) | (results_df['p_logrank'] < 0.05)]
    if len(sig) > 0:
        sig_genes = sig['gene'].tolist()
        sig.to_csv(output_dir / 'transcriptomics_survival_sig_genes.csv', index=False)
        print(f"\n✅ 显著基因 (p_cox<0.05 或 p_logrank<0.05): {len(sig)} 个")
        print(sig[['gene', 'tissue_origin', 'HR', 'p_cox', 'p_logrank']].head(30).to_string(index=False))
        print(f"\n显著基因列表已保存: {output_dir / 'transcriptomics_survival_sig_genes.csv'}")
    else:
        print("\n无显著基因")


if __name__ == '__main__':
    main()
