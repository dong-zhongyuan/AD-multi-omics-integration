#!/usr/bin/env python3
"""
差异表达一致性分析：验证iNPH DEG能否代表AD DEG

对比：
  1. iNPH CSF DEG (High vs Low MOCA) vs AD Brain DEG (AD vs Control, GSE140841)
  2. iNPH PBMC DEG (High vs Low MOCA) vs AD Blood DEG (AD vs HC, GSE226602)

分析指标：
  - 基因overlap（显著DEG的交集）
  - 方向一致性（sign(logFC)一致性比例，二项检验）
  - 效应量相关性（Pearson/Spearman）
  - Hub基因和验证边基因的特别关注
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output/step6_external_validation/deg_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载iNPH DEG数据
# ============================================================
print("=" * 60)
print("1. 加载iNPH DEG数据")
print("=" * 60)

inph_deg = pd.read_csv(PROJECT_ROOT / "processed-data/transcriptomics_deg_genes.csv")
print(f"  iNPH DEG基因: {len(inph_deg)}")
print(f"  列: {list(inph_deg.columns)}")
print(f"  brain_logfc范围: [{inph_deg['brain_logfc'].min():.2f}, {inph_deg['brain_logfc'].max():.2f}]")
print(f"  blood_logfc范围: [{inph_deg['blood_logfc'].min():.2f}, {inph_deg['blood_logfc'].max():.2f}]")

# ============================================================
# 2. 加载并处理GSE140841 (AD Brain)
# ============================================================
print("\n" + "=" * 60)
print("2. 加载GSE140841 AD Brain数据")
print("=" * 60)

# 读取TPM矩阵
tpm_df = pd.read_csv(
    PROJECT_ROOT / "processed-data/step6_external_validation/GSE140841_brain_gene_tpm.csv.gz",
    index_col=0
)
meta = pd.read_csv(
    PROJECT_ROOT / "processed-data/step6_external_validation/GSE140841_brain_metadata.csv"
)

print(f"  TPM矩阵: {tpm_df.shape[0]} 基因 × {tpm_df.shape[1]} 样本")
print(f"  Metadata: {len(meta)} 样本")

# 用acc(GSM ID)匹配TPM列名
meta['sample_id'] = meta['acc']  # GSM4188623 etc

# 只保留AD vs Control
meta_f = meta[meta['diagnosis'].isin(['AD', 'Control'])].copy()
ad_samples = meta_f[meta_f['diagnosis'] == 'AD']['sample_id'].tolist()
ctrl_samples = meta_f[meta_f['diagnosis'] == 'Control']['sample_id'].tolist()
print(f"  AD: {len(ad_samples)}, Control: {len(ctrl_samples)}")

# 确保样本在TPM中
ad_samples = [s for s in ad_samples if s in tpm_df.columns]
ctrl_samples = [s for s in ctrl_samples if s in tpm_df.columns]
print(f"  TPM中: AD={len(ad_samples)}, Control={len(ctrl_samples)}")

# log2(TPM+1)转换
tpm_log = np.log2(tpm_df + 1)

# 计算AD vs Control DEG
print("\n  计算AD DEG...")
ad_expr = tpm_log[ad_samples]
ctrl_expr = tpm_log[ctrl_samples]

ad_mean = ad_expr.mean(axis=1)
ctrl_mean = ctrl_expr.mean(axis=1)
logfc = ad_mean - ctrl_mean  # AD - Control

# Welch's t-test
from scipy.stats import ttest_ind
pvals = []
for gene in tpm_log.index:
    if gene in ad_expr.index and gene in ctrl_expr.index:
        stat, p = ttest_ind(ad_expr.loc[gene], ctrl_expr.loc[gene], equal_var=False)
        pvals.append(p)
    else:
        pvals.append(np.nan)

ad_deg = pd.DataFrame({
    'gene_symbol': tpm_log.index,
    'ad_mean': ad_mean.values,
    'ctrl_mean': ctrl_mean.values,
    'logFC': logfc.values,
    'p_value': pvals
})
ad_deg['neg_log10_p'] = -np.log10(ad_deg['p_value'].clip(lower=1e-300))

n_sig = (ad_deg['p_value'] < 0.05).sum()
print(f"  显著DEG (p<0.05): {n_sig}/{len(ad_deg)}")
print(f"  上调 (AD>Ctrl): {(ad_deg['logFC'] > 0).sum()}")
print(f"  下调 (AD<Ctrl): {(ad_deg['logFC'] < 0).sum()}")

# ============================================================
# 3. ENSG → Symbol 映射（用于iNPH DEG）
# ============================================================
print("\n" + "=" * 60)
print("3. ENSG → Symbol 映射")
print("=" * 60)

# 从AD DEG数据中直接获取symbol→ENSG映射不太直接。
# 反过来：从iNPH的ENSG ID转换为symbol，用mygene或已有映射

# 尝试用项目中已有的转换
import sys
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from tools.gene_id_converter import ensembl_to_symbol
    ensg_list = inph_deg['gene_id'].tolist()
    symbol_map = ensembl_to_symbol(ensg_list)
    inph_deg['gene_symbol'] = inph_deg['gene_id'].map(symbol_map)
    mapped = inph_deg['gene_symbol'].notna().sum()
    print(f"  映射成功: {mapped}/{len(inph_deg)}")
except Exception as e:
    print(f"  gene_id_converter失败: {e}")
    print("  尝试备用方案: mygene...")
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        ensg_list = inph_deg['gene_id'].str.replace(r'\.\d+$', '', regex=True).tolist()
        results = mg.querymany(ensg_list, scopes='ensembl.gene', fields='symbol', species='human')
        sym_map = {}
        for r in results:
            if 'symbol' in r:
                sym_map[r['query']] = r['symbol']
        # 也尝试带版本号的
        inph_deg['gene_symbol'] = inph_deg['gene_id'].str.replace(r'\.\d+$', '', regex=True).map(sym_map)
        mapped = inph_deg['gene_symbol'].notna().sum()
        print(f"  mygene映射成功: {mapped}/{len(inph_deg)}")
    except Exception as e2:
        print(f"  mygene也失败: {e2}")
        inph_deg['gene_symbol'] = None

# ============================================================
# 4. 合并对比
# ============================================================
print("\n" + "=" * 60)
print("4. 合并iNPH DEG和AD DEG")
print("=" * 60)

# 基于gene_symbol合并
merged = inph_deg.merge(
    ad_deg[['gene_symbol', 'logFC', 'p_value', 'neg_log10_p']],
    on='gene_symbol',
    how='inner',
    suffixes=('_inph', '_ad')
)
print(f"  共通基因: {len(merged)}/{len(inph_deg)} ({(len(merged)/len(inph_deg)*100):.1f}%)")

if len(merged) == 0:
    print("  ❌ 没有共通基因！检查ID映射问题。")
    # 不退出，继续输出调试信息
    print(f"  iNPH gene_symbol样例: {inph_deg['gene_symbol'].dropna().head(10).tolist()}")
    print(f"  AD gene_symbol样例: {ad_deg['gene_symbol'].head(10).tolist()}")

# ============================================================
# 5. 一致性分析
# ============================================================
if len(merged) > 0:
    print("\n" + "=" * 60)
    print("5. 一致性分析")
    print("=" * 60)

    # 方向一致性: iNPH brain_logfc vs AD brain logFC
    inph_dir = np.sign(merged['brain_logfc'])
    ad_dir = np.sign(merged['logFC'])
    
    # 排除任一为0的
    valid = (inph_dir != 0) & (ad_dir != 0)
    n_valid = valid.sum()
    n_consistent = (inph_dir[valid] == ad_dir[valid]).sum()
    
    print(f"\n  【方向一致性: iNPH CSF vs AD Brain】")
    print(f"  有效基因(非零logFC): {n_valid}")
    print(f"  方向一致: {n_consistent}/{n_valid} ({n_consistent/n_valid*100:.1f}%)")
    
    if n_valid > 0:
        binom_p = binomtest(n_consistent, n_valid, p=0.5, alternative='greater').pvalue
        print(f"  二项检验 p = {binom_p:.4f}")
    
    # 效应量相关性
    r_pearson, p_pearson = stats.pearsonr(merged['brain_logfc'], merged['logFC'])
    r_spearman, p_spearman = stats.spearmanr(merged['brain_logfc'], merged['logFC'])
    print(f"\n  【效应量相关性】")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    print(f"  Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.4f}")
    
    # 只看显著DEG
    sig_inph = merged['brain_logfc'].abs() > np.percentile(merged['brain_logfc'].abs(), 80)  # top 20%
    sig_ad = merged['p_value'] < 0.05
    sig_both = sig_inph & sig_ad
    
    inph_dir_sig = np.sign(merged.loc[sig_both, 'brain_logfc'])
    ad_dir_sig = np.sign(merged.loc[sig_both, 'logFC'])
    valid_sig = (inph_dir_sig != 0) & (ad_dir_sig != 0)
    
    if valid_sig.sum() > 0:
        n_cons_sig = (inph_dir_sig[valid_sig] == ad_dir_sig[valid_sig]).sum()
        print(f"\n  【显著DEG方向一致性 (iNPH top20% & AD p<0.05)】")
        print(f"  基因数: {valid_sig.sum()}")
        print(f"  方向一致: {n_cons_sig}/{valid_sig.sum()} ({n_cons_sig/valid_sig.sum()*100:.1f}%)")

    # ============================================================
    # 6. Hub基因和验证边基因的特别分析
    # ============================================================
    print("\n" + "=" * 60)
    print("6. Hub基因和验证边基因分析")
    print("=" * 60)
    
    # 加载验证边
    edges = pd.read_csv(PROJECT_ROOT / "output/verified_cross_tissue_edges.csv")
    
    # 提取所有涉及的基因symbol
    edge_genes = set(edges['source'].tolist() + edges['target'].tolist())
    print(f"  验证边涉及基因: {len(edge_genes)}")
    
    # 标记
    merged['is_edge_gene'] = merged['gene_symbol'].isin(edge_genes)
    edge_merged = merged[merged['is_edge_gene']]
    print(f"  在共通基因中的边基因: {len(edge_merged)}")
    
    if len(edge_merged) > 0:
        edge_inph_dir = np.sign(edge_merged['brain_logfc'])
        edge_ad_dir = np.sign(edge_merged['logFC'])
        edge_valid = (edge_inph_dir != 0) & (edge_ad_dir != 0)
        
        if edge_valid.sum() > 0:
            edge_cons = (edge_inph_dir[edge_valid] == edge_ad_dir[edge_valid]).sum()
            print(f"  边基因方向一致: {edge_cons}/{edge_valid.sum()} ({edge_cons/edge_valid.sum()*100:.1f}%)")

    # ============================================================
    # 7. 保存结果
    # ============================================================
    print("\n" + "=" * 60)
    print("7. 保存结果")
    print("=" * 60)
    
    merged.to_csv(OUTPUT_DIR / "inph_vs_ad_deg_comparison.csv", index=False)
    print(f"  已保存: {OUTPUT_DIR / 'inph_vs_ad_deg_comparison.csv'}")
    
    # ============================================================
    # 8. 可视化
    # ============================================================
    print("\n" + "=" * 60)
    print("8. 生成图表")
    print("=" * 60)
    
    plt.rcParams['figure.dpi'] = 150
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 8.1 散点图: iNPH logFC vs AD logFC (全基因)
    ax = axes[0, 0]
    ax.scatter(merged['brain_logfc'], merged['logFC'], 
              alpha=0.15, s=3, c='steelblue', edgecolors='none')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('iNPH CSF logFC (Low vs High MOCA)')
    ax.set_ylabel('AD Brain logFC (AD vs Control)')
    ax.set_title(f'All {len(merged)} genes\nPearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}')
    
    # 添加回归线
    z = np.polyfit(merged['brain_logfc'], merged['logFC'], 1)
    x_line = np.linspace(merged['brain_logfc'].min(), merged['brain_logfc'].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r-', alpha=0.5, linewidth=1)
    
    # 8.2 散点图: 边基因高亮
    ax = axes[0, 1]
    non_edge = merged[~merged['is_edge_gene']]
    ax.scatter(non_edge['brain_logfc'], non_edge['logFC'], 
              alpha=0.1, s=3, c='lightgray', edgecolors='none', label='Other')
    ax.scatter(edge_merged['brain_logfc'], edge_merged['logFC'], 
              alpha=0.8, s=30, c='red', edgecolors='white', linewidth=0.5, label='Edge genes')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('iNPH CSF logFC')
    ax.set_ylabel('AD Brain logFC')
    ax.set_title(f'Edge genes highlighted ({len(edge_merged)} genes)')
    ax.legend(fontsize=8)
    
    # 8.3 方向一致性饼图
    ax = axes[0, 2]
    if n_valid > 0:
        sizes = [n_consistent, n_valid - n_consistent]
        labels = [f'Consistent\n({n_consistent})', f'Discordant\n({n_valid - n_consistent})']
        colors = ['#4CAF50', '#F44336']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Direction Consistency\np={binom_p:.4f}')
    
    # 8.4 logFC分布对比
    ax = axes[1, 0]
    inph_fc = merged['brain_logfc'].dropna()
    ad_fc = merged['logFC'].dropna()
    if len(inph_fc) > 0 and len(ad_fc) > 0:
        ax.hist(inph_fc, bins=80, alpha=0.5, label='iNPH CSF', color='steelblue', density=True)
        ax.hist(ad_fc, bins=80, alpha=0.5, label='AD Brain', color='coral', density=True)
    ax.set_xlabel('logFC')
    ax.set_ylabel('Density')
    ax.set_title('logFC Distribution Comparison')
    ax.legend()
    
    # 8.5 按iNPH logFC分bin看AD logFC一致性
    ax = axes[1, 1]
    # 分10个bin
    merged['logFC_bin'] = pd.qcut(merged['brain_logfc'], q=10, labels=False, duplicates='drop')
    bin_stats = merged.groupby('logFC_bin').agg(
        mean_inph=('brain_logfc', 'mean'),
        mean_ad=('logFC', 'mean'),
        sem_ad=('logFC', 'sem'),
        n=('logFC', 'count')
    ).reset_index()
    
    ax.errorbar(bin_stats['mean_inph'], bin_stats['mean_ad'], 
               yerr=bin_stats['sem_ad'], fmt='o-', capsize=4, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('iNPH CSF mean logFC (per decile)')
    ax.set_ylabel('AD Brain mean logFC (per decile)')
    ax.set_title('Binned Comparison (±SEM)')
    
    # 8.6 显著DEG overlap (Venn图用文字替代)
    ax = axes[1, 2]
    ax.axis('off')
    
    # 统计
    sig_inph_count = sig_inph.sum()
    sig_ad_count = sig_ad.sum()
    sig_both_count = sig_both.sum()
    inph_only = sig_inph_count - sig_both_count
    ad_only = sig_ad_count - sig_both_count
    
    summary_text = f"""Gene Overlap Summary
    ─────────────────
    iNPH top20% DEG: {sig_inph_count}
    AD p<0.05 DEG:    {sig_ad_count}
    Overlap:          {sig_both_count}
    
    iNPH only:        {inph_only}
    AD only:          {ad_only}
    
    Direction consistency
    in overlap:       {n_cons_sig if valid_sig.sum() > 0 else 'N/A'}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
           fontsize=10, fontfamily='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inph_vs_ad_deg_consistency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {OUTPUT_DIR / 'inph_vs_ad_deg_consistency.png'}")
    
    # ============================================================
    # 9. 总结报告
    # ============================================================
    print("\n" + "=" * 60)
    print("9. 生成报告")
    print("=" * 60)
    
    verdict = "✅ iNPH DEG方向与AD DEG显著一致" if (n_valid > 0 and binom_p < 0.05) else \
              "⚠️ 方向一致性未达显著水平"
    
    report = f"""# 差异表达一致性分析：iNPH vs AD

## 数据
- **iNPH**: GSE292141 (单细胞, CSF+PBMC), High vs Low MOCA, {len(inph_deg)} DEGs
- **AD Brain**: GSE140841 (bulk RNA-seq, BA9+EC), AD vs Control, {len(ad_deg)} genes tested
- **共通基因**: {len(merged)} ({len(merged)/len(inph_deg)*100:.1f}%)

## 方向一致性
- 有效基因: {n_valid}
- 方向一致: {n_consistent}/{n_valid} ({n_consistent/n_valid*100:.1f}%)
- 二项检验: p = {binom_p:.4f}

## 效应量相关性
- Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f})
- Spearman ρ = {r_spearman:.4f} (p = {p_spearman:.4f})

## 显著DEG overlap
- iNPH top20%: {sig_inph_count}
- AD p<0.05: {sig_ad_count}
- Overlap: {sig_both_count}
- 重叠中方向一致: {n_cons_sig if valid_sig.sum() > 0 else 'N/A'}

## 验证边基因
- 边基因数: {len(edge_genes)}
- 在共通基因中: {len(edge_merged)}
{(f'- 方向一致: {edge_cons}/{edge_valid.sum()} ({edge_cons/edge_valid.sum()*100:.1f}%)' if len(edge_merged) > 0 and edge_valid.sum() > 0 else '')}

## 结论
{verdict}
"""
    
    with open(OUTPUT_DIR / "report.md", 'w') as f:
        f.write(report)
    print(report)
    print(f"\n  报告已保存: {OUTPUT_DIR / 'report.md'}")

else:
    print("\n❌ 没有共通基因，无法进行一致性分析。")
    print("调试信息:")
    print(f"  iNPH gene_symbol前10: {inph_deg['gene_symbol'].dropna().head(10).tolist()}")
    print(f"  AD gene_symbol前10: {ad_deg['gene_symbol'].head(10).tolist()}")
    
    # 尝试直接匹配
    inph_syms = set(inph_deg['gene_symbol'].dropna())
    ad_syms = set(ad_deg['gene_symbol'].dropna())
    intersection = inph_syms & ad_syms
    print(f"  集合交集: {len(intersection)}")
    if len(intersection) > 0:
        print(f"  前10: {list(intersection)[:10]}")

print("\n" + "=" * 60)
print("分析完成!")
