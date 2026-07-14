#!/usr/bin/env python3
"""
分析脑组织的Eigengene并检查与Hub基因的交集

Eigengene识别方法：
1. WGCNA (Weighted Gene Co-expression Network Analysis)
2. PCA (Principal Component Analysis) - 每个模块的第一主成分
3. 基于网络中心性的方法
"""

import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")
import networkx as nx

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config
config = get_config()

# 路径配置
STEP2_DIR = PROJECT_ROOT / "output/step2_cross_tissue_causality"
STEP3_DIR = PROJECT_ROOT / "output/step3_hub_identification"
OUTPUT_DIR = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OMICS_TYPES = ['proteomics','transcriptomics','metabolomics']  


def map_protein_to_gene(protein_name):
    """
    映射蛋白质名称到基因symbol
    规则：
    - BD-XXX → XXX（去除血液来源前缀）
    - pTau-XXX → MAPT（磷酸化Tau蛋白）
    - Aβ/AÎ²/A? → APP（淀粉样蛋白前体）
    - 其他 → 保持原样
    """
    if pd.isna(protein_name):
        return protein_name
    
    protein_name = str(protein_name)
    
    # BD-前缀：血液来源的蛋白质
    if protein_name.startswith('BD-'):
        base_name = protein_name[3:]  # 去除"BD-"
        # pTau系列都映射到MAPT
        if base_name.startswith('pTau'):
            return 'MAPT'
        return base_name
    
    # pTau系列（无BD-前缀）
    if protein_name.startswith('pTau'):
        return 'MAPT'
    
    # Aβ系列（淀粉样蛋白β）-> APP基因
    if protein_name.startswith('Aβ') or protein_name.startswith('AÎ²') or protein_name.startswith('A?'):
        return 'APP'
    
    return protein_name


def merge_duplicate_edges_stouffer(edges_df):
    """
    使用Stouffer's method合并重复边（相同source-target对）
    
    Args:
        edges_df: 包含source, target, weight等列的DataFrame
    
    Returns:
        merged_df: 去重后的DataFrame，统计量已合并
    """
    from scipy import stats
    
    print(f"  原始边数: {len(edges_df)}")
    
    # 按(source, target)分组
    grouped = edges_df.groupby(['source', 'target'])
    
    merged_rows = []
    for (source, target), group in grouped:
        if len(group) == 1:
            # 单条边，直接保留
            merged_rows.append(group.iloc[0].to_dict())
        else:
            # 多条边，需要合并
            # 使用Stouffer's method合并：z_combined = sum(z_i) / sqrt(n)
            # weight和strength视为效应量，合并后应该更强
            weights = group['weight'].values
            strengths = group['strength'].values
            n = len(group)
            
            # Stouffer's method: 合并后的z-score = sum(z_i) / sqrt(n)
            # 这里weight/strength本身就是效应量，直接求和后除以sqrt(n)
            merged_weight = np.sum(weights) / np.sqrt(n)
            merged_strength = np.sum(strengths) / np.sqrt(n)
            
            # 保留第一条边的其他信息
            merged_row = group.iloc[0].to_dict()
            merged_row['weight'] = merged_weight
            merged_row['strength'] = merged_strength
            merged_row['n_merged'] = n  # 记录合并了几条边
            
            merged_rows.append(merged_row)
    
    merged_df = pd.DataFrame(merged_rows)
    n_merged = len(edges_df) - len(merged_df)
    if n_merged > 0:
        print(f"  合并了 {n_merged} 条重复边")
        print(f"  去重后边数: {len(merged_df)}")
    
    return merged_df  


def load_network(omics_type, tissue):
    """加载网络数据"""
    edges_file = STEP2_DIR / omics_type / f"{tissue}_network" / "consensus_edges.csv"
    if not edges_file.exists():
        raise FileNotFoundError(f"找不到网络文件: {edges_file}")
    
    edges = pd.read_csv(edges_file)
    print(f"  原始{tissue}内边: {len(edges)} 条")
    
    # 排除自环边（source == target）
    edges = edges[edges['source'] != edges['target']]
    print(f"  排除自环边后: {len(edges)} 条")
    
    return edges


def load_brain_network(omics_type):
    """加载脑组织内网络"""
    return load_network(omics_type, "brain")


def identify_disease_associated_genes(omics_type, tissue, pvalue_threshold=0.05):
    """
    识别与AD疾病相关的基因（基于Gene Significance）
    
    使用 |Spearman ρ| > 0.1 的效果量阈值筛选（WGCNA 原文定义）。
    大样本下 p 值失去区分力，效果量阈值才是有意义的筛子。
    
    Args:
        omics_type: 组学类型 (proteomics, transcriptomics, metabolomics)
        tissue: 组织类型 (brain/csf, blood/plasma)
        pvalue_threshold: 仅用于报告，不参与筛选
    
    Returns:
        disease_genes: 与疾病显著相关的基因列表
        results_df: 详细结果DataFrame
    """
    import anndata as ad
    from scipy.stats import spearmanr
    
    print(f"\\n[Gene Significance] 识别与AD相关的{tissue}端基因...")
    
    # 1. 加载表达数据
    if omics_type == 'proteomics':
        if tissue == 'brain':
            adata_file = PROJECT_ROOT / 'processed-data/csf_proteomics_paired.h5ad'
        else:
            adata_file = PROJECT_ROOT / 'processed-data/plasma_proteomics_paired.h5ad'
    elif omics_type == 'transcriptomics':
        if tissue == 'brain':
            adata_file = PROJECT_ROOT / 'processed-data/transcriptomics_brain.h5ad'
        else:
            adata_file = PROJECT_ROOT / 'processed-data/transcriptomics_blood.h5ad'
    elif omics_type == 'metabolomics':
        if tissue == 'brain':
            adata_file = PROJECT_ROOT / 'processed-data/csf_metabolomics_common.h5ad'
        else:
            adata_file = PROJECT_ROOT / 'processed-data/plasma_metabolomics_common.h5ad'
    else:
        raise ValueError(f"Unknown omics type: {omics_type}")
    
    if not adata_file.exists():
        print(f"  ✗ 找不到表达数据文件: {adata_file}")
        return [], pd.DataFrame()
    
    adata = ad.read_h5ad(adata_file)
    print(f"  加载表达数据: {adata.n_obs} 样本 × {adata.n_vars} 基因")
    
    # 2. 加载诊断信息（根据组学类型选择不同的诊断文件）
    if omics_type == 'transcriptomics':
        # 5xFAD: 直接用 obs 里的 genotype 列（WT=control, 5xFAD=case）
        if 'genotype' in adata.obs.columns:
            print(f"  使用 genotype 列作为诊断变量（5xFAD 模式）")
            merged = adata.obs.copy()
            merged['DIAGNOSIS'] = merged['genotype'].map({'WT': 1.0, '5xFAD': 3.0})

            # 单细胞水平 GS：每个细胞的 genotype vs 该基因表达
            gene_names = adata.var_names.astype(str).tolist()
            diagnosis = merged['DIAGNOSIS'].to_numpy()
            valid = ~np.isnan(diagnosis)

            import scipy.sparse as sp_mod
            X_dense = adata.X.toarray() if sp_mod.issparse(adata.X) else np.array(adata.X)
            X_valid = X_dense[valid]  # (n_valid_cells, n_genes)
            diag_valid = diagnosis[valid]

            significant_genes = []
            gs_results = []
            for gi, gene in enumerate(gene_names):
                gene_expr = X_valid[:, gi]
                if np.std(gene_expr) < 1e-8:
                    corr, pval = 0.0, 1.0
                else:
                    corr, pval = spearmanr(gene_expr, diag_valid)
                    if np.isnan(corr):
                        corr, pval = 0.0, 1.0
                if pval < 0.1:
                    significant_genes.append(gene)
                gs_results.append({'gene': gene, 'correlation': corr, 'pvalue': pval})
            gs_df = pd.DataFrame(gs_results)
            print(f"  显著相关基因 (p<0.05): {len(significant_genes)}/{len(gene_names)}")
            return significant_genes, gs_df

        # 血端 genotype=mixed 时无法做 GS（BM 是 WT+5xFAD 混合）
        # 此时用全部血端基因作为疾病相关（不做 GS 筛选）
        if 'genotype' in adata.obs.columns and adata.obs['genotype'].nunique() == 1:
            print(f"  血端 genotype 不可分组（混合样本），跳过 GS，保留全部基因")
            gene_names = adata.var_names.astype(str).tolist()
            return gene_names, pd.DataFrame({'gene': gene_names, 'correlation': 0.0, 'pvalue': 1.0})
        else:
            # 旧版 iNPH 模式（保留兼容）
            if tissue == 'brain':
                dx_file = PROJECT_ROOT / 'processed-data/transcriptomics_brain_sample_diagnosis.tsv'
                if not dx_file.exists():
                    print(f"  ✗ 找不到脑转录组诊断文件: {dx_file}")
                    return [], pd.DataFrame()
                dx = pd.read_csv(dx_file, sep='\t')
            else:
                dx_file = PROJECT_ROOT / 'processed-data/transcriptomics_blood_sample_diagnosis.tsv'
                if not dx_file.exists():
                    print(f"  ✗ 找不到血液转录组诊断文件: {dx_file}")
                    return [], pd.DataFrame()
                dx = pd.read_csv(dx_file, sep='\t')

            if 'diagnosis_numeric' in dx.columns:
                dx['DIAGNOSIS'] = dx['diagnosis_numeric']
            elif 'DIAGNOSIS' not in dx.columns:
                print(f"  ✗ 转录组诊断文件缺少 diagnosis_numeric/DIAGNOSIS 列: {dx_file}")
                return [], pd.DataFrame()
    else:
        # 蛋白质组和代谢组使用ADNI诊断
        # 但代谢组数据已经包含诊断信息在obs['diagnosis']中
        if omics_type == 'metabolomics' and 'diagnosis' in adata.obs.columns:
            # 代谢组直接使用obs中的诊断信息
            diagnosis_map = {'AD': 3.0, 'MCI': 2.0, 'CN': 1.0}
            merged = adata.obs.copy()
            merged['DIAGNOSIS'] = merged['diagnosis'].map(diagnosis_map)
        else:
            # 蛋白质组使用ADNI诊断文件
            dx_file = PROJECT_ROOT / 'data/blood-transcription-protein/DXSUM_17Apr2026.csv'
            dx = pd.read_csv(dx_file)
    
    # 3. 匹配诊断信息
    if omics_type == 'transcriptomics' and 'sample_id' in adata.obs.columns:
        sample_ids = adata.obs['sample_id'].astype(str).tolist()
    else:
        sample_ids = adata.obs.index.tolist()
    
    if omics_type == 'transcriptomics':
        # 转录组直接用sample_id匹配
        merged = pd.DataFrame({'sample_id': sample_ids})
        merged = merged.merge(
            dx[['sample_id', 'DIAGNOSIS']],
            on='sample_id',
            how='left'
        )
    elif omics_type == 'metabolomics' and 'DIAGNOSIS' in merged.columns:
        # 代谢组已经在上面处理好了merged，直接跳过
        pass
    else:
        # 蛋白质组和代谢组用PTID+VISCODE匹配
        ptids = []
        viscodes = []
        for sid in sample_ids:
            if '_' in sid:
                parts = sid.rsplit('_', 1)
                if len(parts) == 2:
                    ptids.append(parts[0])
                    viscodes.append(parts[1])
                else:
                    ptids.append(sid)
                    viscodes.append('bl')
            else:
                ptids.append(sid)
                viscodes.append('bl')
        
        adata.obs['PTID'] = ptids
        adata.obs['VISCODE'] = viscodes
        
        merged = adata.obs.merge(
            dx[['PTID', 'VISCODE', 'DIAGNOSIS']],
            on=['PTID', 'VISCODE'],
            how='left'
        )
    
    if omics_type == 'transcriptomics':
        # 转录组改为病人级 pseudobulk：先按 sample_id 聚合细胞，再做 GS
        print(f"  使用病人级 pseudobulk 统计...")
        sample_ids_array = adata.obs['sample_id'].astype(str).to_numpy()
        dx_unique = dx[['sample_id', 'DIAGNOSIS', 'diagnosis']].drop_duplicates()

        pseudobulk_rows = []
        pseudobulk_diagnosis = []
        kept_sample_ids = []
        for _, row in dx_unique.iterrows():
            sid = str(row['sample_id'])
            cell_mask = sample_ids_array == sid
            if not cell_mask.any():
                continue
            sample_expr = np.asarray(adata.X[cell_mask, :].mean(axis=0)).ravel()
            pseudobulk_rows.append(sample_expr)
            pseudobulk_diagnosis.append(row['DIAGNOSIS'])
            kept_sample_ids.append(sid)

        n_valid = len(pseudobulk_rows)
        if n_valid < 4:
            print(f"  ✗ 有效病人级样本数太少: {n_valid}")
            return [], pd.DataFrame()

        print(f"  成功匹配诊断: {n_valid}/{len(dx_unique)} 病人级样本")
        diagnosis = np.asarray(pseudobulk_diagnosis, dtype=np.float32)
        expr_data = np.vstack(pseudobulk_rows).astype(np.float32)
    else:
        # 只保留有诊断的样本
        valid_mask = merged['DIAGNOSIS'].notna().to_numpy()
        n_valid = valid_mask.sum()
        
        if n_valid < 10:
            print(f"  ✗ 有效样本数太少: {n_valid}")
            print(f"  ⚠️  该组学数据无ADNI诊断标签，无法使用Gene Significance方法")
            return [], pd.DataFrame()
        
        print(f"  成功匹配诊断: {n_valid}/{len(sample_ids)} 样本")
        
        diagnosis = merged.loc[valid_mask, 'DIAGNOSIS'].values
        expr_data = adata.X[valid_mask, :]

    if pd.Series(diagnosis).nunique() < 2:
        if omics_type == 'transcriptomics' and 'diagnosis' in dx.columns:
            labels = dx[['sample_id', 'diagnosis']].drop_duplicates()['diagnosis'].value_counts().to_dict()
            print(f"  ✗ 诊断标签只有一个类别，无法做疾病相关性分析: {labels}")
        else:
            print(f"  ✗ 诊断标签只有一个类别，无法做疾病相关性分析")
        return [], pd.DataFrame()

    # 诊断分布
    unique, counts = np.unique(diagnosis, return_counts=True)
    if omics_type == 'transcriptomics' and 'diagnosis' in dx.columns:
        print(f"  诊断分布: {dx[['sample_id', 'diagnosis']].drop_duplicates()['diagnosis'].value_counts().to_dict()}")
    else:
        print(f"  诊断分布: CN={counts[unique==1.0][0] if 1.0 in unique else 0}, "
              f"MCI={counts[unique==2.0][0] if 2.0 in unique else 0}, "
              f"AD={counts[unique==3.0][0] if 3.0 in unique else 0}")
    
    # 4. 计算每个基因与DIAGNOSIS的Spearman相关
    gene_names = adata.var.index.tolist()
    correlations = []
    pvalues = []
    
    for i in range(adata.n_vars):
        gene_expr = expr_data[:, i].toarray().flatten() if hasattr(expr_data, 'toarray') else np.asarray(expr_data[:, i]).ravel()
        
        # 去除NaN
        valid = ~np.isnan(gene_expr)
        if valid.sum() < 10:
            correlations.append(np.nan)
            pvalues.append(np.nan)
            continue
        
        corr, pval = spearmanr(gene_expr[valid], diagnosis[valid])
        correlations.append(corr)
        pvalues.append(pval)
    
    # 5. 创建结果DataFrame
    results_df = pd.DataFrame({
        'gene': gene_names,
        'correlation': correlations,
        'pvalue': pvalues,
        'abs_correlation': np.abs(correlations)
    })
    
    # 排序
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    
    # 6. 选择显著相关的基因（p<0.1 探索性筛选阈值）
    disease_genes = results_df[results_df['pvalue'] < 0.1]['gene'].tolist()

    print(f"  疾病相关基因 (p<0.1): {len(disease_genes)}/{len(gene_names)} ({len(disease_genes)/max(len(gene_names),1)*100:.1f}%)")
    print(f"  Top 5: {disease_genes[:5]}")
    
    return disease_genes, results_df


def identify_eigengenes_by_modules(edges_df, top_n_per_module=10):
    """
    方法2：基于社区检测识别模块，然后找每个模块的Eigengene
    
    使用Louvain算法检测社区（模块），每个模块选择度最高的基因作为Eigengene
    """
    print("\n[方法2] 基于模块检测识别Eigengene...")
    
    # 构建无向图（社区检测需要无向图）
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        if G.has_edge(row['source'], row['target']):
            G[row['source']][row['target']]['weight'] += row['strength']
        else:
            G.add_edge(row['source'], row['target'], weight=row['strength'])
    
    print(f"  网络节点数: {G.number_of_nodes()}")
    print(f"  网络边数: {G.number_of_edges()}")
    
    # Louvain社区检测
    print("  运行Louvain社区检测...")
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight='weight')
    except ImportError:
        print("  python-louvain未安装，使用贪心模块度算法")
        from networkx.algorithms import community as nx_community
        communities = nx_community.greedy_modularity_communities(G, weight='weight')
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
    
    # 统计模块
    modules = {}
    for node, module_id in partition.items():
        if module_id not in modules:
            modules[module_id] = []
        modules[module_id].append(node)
    
    print(f"  检测到 {len(modules)} 个模块")
    
    # 每个模块选择度最高的基因作为Eigengene
    eigengenes = []
    module_info = []
    
    for module_id, genes in modules.items():
        if len(genes) < 5:  # 跳过太小的模块
            continue
        
        # 计算模块内每个基因的度
        subgraph = G.subgraph(genes)
        degrees = dict(subgraph.degree(weight='weight'))
        
        # 选择top N
        sorted_genes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        module_eigengenes = [gene for gene, deg in sorted_genes[:top_n_per_module]]
        eigengenes.extend(module_eigengenes)
        
        module_info.append({
            'module_id': module_id,
            'module_size': len(genes),
            'eigengenes': module_eigengenes,
            'top_eigengene': module_eigengenes[0] if module_eigengenes else None,
            'top_degree': sorted_genes[0][1] if sorted_genes else 0,
        })
    
    # 保存模块信息
    if module_info:
        module_df = pd.DataFrame(module_info)
        print(f"  识别出 {len(eigengenes)} 个Eigengene（来自 {len(module_info)} 个模块）")
        print(f"  模块大小范围: {module_df['module_size'].min()} - {module_df['module_size'].max()}")
    else:
        module_df = pd.DataFrame()
        print(f"  警告：未找到有效模块（模块太小或网络太稀疏）")
    
    return eigengenes, module_df


def compare_with_hubs(eigengenes, hub_genes):
    """比较Eigengene和Hub基因的交集"""
    eigengene_set = set(eigengenes)
    hub_set = set(hub_genes)
    
    overlap = eigengene_set & hub_set
    only_eigengene = eigengene_set - hub_set
    only_hub = hub_set - eigengene_set
    
    print(f"\n[交集分析]")
    print(f"  Eigengene数量: {len(eigengene_set)}")
    print(f"  Hub基因数量: {len(hub_set)}")
    
    if len(eigengene_set) > 0:
        print(f"  交集: {len(overlap)} 个基因 ({len(overlap)/len(eigengene_set)*100:.1f}% of Eigengenes)")
    else:
        print(f"  交集: {len(overlap)} 个基因 (N/A - 无Eigengene)")
    
    print(f"  仅Eigengene: {len(only_eigengene)} 个")
    print(f"  仅Hub: {len(only_hub)} 个")
    
    if overlap:
        print(f"\n  交集基因（前10个）: {list(overlap)[:10]}")
    
    return {
        'overlap': list(overlap),
        'only_eigengene': list(only_eigengene),
        'only_hub': list(only_hub),
        'overlap_ratio': len(overlap) / len(eigengene_set) if eigengene_set else 0,
    }

def filter_edges_by_elbow(edges_df, window_size=10, max_drop_threshold=0.05, min_edges=None):
    """
    基于骤降点筛选边（不使用血端表达量）
    
    Args:
        edges_df: 边的DataFrame
        window_size: 滑动窗口大小
        max_drop_threshold: 最大骤降率阈值（默认5%）
        min_edges: 最少保留边数（如果设置，则至少保留这么多边）
    
    Returns:
        filtered_df: 筛选后的边
        elbow_rank: 骤降点排名
        drop_analysis: 骤降分析数据
    """
    print(f"\n  [骤降点筛选（含血端表达量评分）]")
    if min_edges is not None:
        print(f"    参数: window_size={window_size}, max_drop_threshold={max_drop_threshold:.1%}, min_edges={min_edges}")
    else:
        print(f"    参数: window_size={window_size}, max_drop_threshold={max_drop_threshold:.1%}")
    
    # 1. 标准化三个置信度指标到[0,1]
    for metric in ['confidence_stability', 'confidence_snr', 'confidence_consistency']:
        if metric not in edges_df.columns:
            print(f"    警告: 缺少{metric}列，跳过骤降点筛选")
            return edges_df, None, None
    
    if 'strength' not in edges_df.columns:
        print(f"    警告: 缺少strength列，跳过骤降点筛选")
        return edges_df, None, None
    
    df = edges_df.copy()
    
    # Min-Max标准化
    df['stability_norm'] = (df['confidence_stability'] - df['confidence_stability'].min()) / \
                           (df['confidence_stability'].max() - df['confidence_stability'].min() + 1e-8)
    df['snr_norm'] = (df['confidence_snr'] - df['confidence_snr'].min()) / \
                     (df['confidence_snr'].max() - df['confidence_snr'].min() + 1e-8)
    df['consistency_norm'] = (df['confidence_consistency'] - df['confidence_consistency'].min()) / \
                             (df['confidence_consistency'].max() - df['confidence_consistency'].min() + 1e-8)
    
    # 2. 计算综合得分（三指标算术平均）
    df['confidence_combined'] = (
        df['stability_norm'] + 
        df['snr_norm'] + 
        df['consistency_norm']
    ) / 3.0
    
    # 3. 标准化strength到[0,1]
    df['strength_norm'] = (np.abs(df['strength']) - np.abs(df['strength']).min()) / \
                          (np.abs(df['strength']).max() - np.abs(df['strength']).min() + 1e-8)
    
    # 4. 计算最终评分：基于虚拟敲除验证的公式
    # 去掉consistency（负相关），只用stability和snr
    df['confidence_ko_optimized'] = 0.5 * df['stability_norm'] + 0.5 * df['snr_norm']
    df['final_score'] = df['strength_norm'] * df['confidence_ko_optimized']
    print(f"    评分公式: final_score = strength × (0.5×snr + 0.5×stability)")
    
    # 6. 按最终评分排序
    df_sorted = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1
    
    print(f"    置信度范围: [{df_sorted['confidence_combined'].min():.3f}, {df_sorted['confidence_combined'].max():.3f}]")
    print(f"    强度范围: [{df_sorted['strength'].min():.6f}, {df_sorted['strength'].max():.6f}]")
    print(f"    最终评分范围: [{df_sorted['final_score'].min():.3f}, {df_sorted['final_score'].max():.3f}]")
    
    # 7. 使用累积占比方法（替代骤降点检测）
    # 计算累积占比
    total_score = df_sorted['final_score'].sum()
    df_sorted['cumsum'] = df_sorted['final_score'].cumsum()
    df_sorted['cumsum_ratio'] = df_sorted['cumsum'] / total_score
    
    # 使用累积90%阈值
    cumulative_threshold = 0.9
    n_edges = (df_sorted['cumsum_ratio'] <= cumulative_threshold).sum()
    if n_edges == 0:
        n_edges = 1  # 至少保留1条
    
    elbow_rank = n_edges
    filtered_df = df_sorted.iloc[:n_edges].copy()
    
    print(f"    累积占比方法: 保留累积{cumulative_threshold:.0%}的边")
    print(f"    截断至Top {n_edges} ({n_edges/len(df_sorted):.1%})" if len(df_sorted) > 0 else "    无边可筛选")
    if len(filtered_df) > 0:
        print(f"    累积分数占比: {filtered_df['cumsum_ratio'].iloc[-1]:.1%}")
    else:
        print("    无边，跳过")
    
    # 保留骤降分析数据（用于调试）
    drop_data = []
    prev_mean = None
    for i in range(0, min(200, len(df_sorted)), window_size):
        window = df_sorted.iloc[i:i+window_size]
        mean_val = window['final_score'].mean()
        
        if prev_mean is not None:
            drop = prev_mean - mean_val
            drop_rate = drop / prev_mean if prev_mean > 0 else 0
            
            drop_data.append({
                'start_rank': i+1,
                'end_rank': i+window_size,
                'mean_score': mean_val,
                'drop': drop,
                'drop_rate': drop_rate
            })
        else:
            drop_data.append({
                'start_rank': i+1,
                'end_rank': i+window_size,
                'mean_score': mean_val,
                'drop': 0,
                'drop_rate': 0
            })
        
        prev_mean = mean_val
    
    return filtered_df, elbow_rank, drop_data


def main():
    print("="*60)
    print("疾病相关基因分析 - 多组织 + 边筛选（Gene Significance方法）")
    print("="*60)
    
    for omics_type in OMICS_TYPES:
        print(f"\n{'='*60}")
        print(f"处理: {omics_type}")
        print(f"{'='*60}")
        
        # 1. 加载脑网络
        print("\n[1] 加载脑组织网络...")
        try:
            brain_edges = load_brain_network(omics_type)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue
        
        # 2. 加载血液网络
        print("\n[2] 加载血液网络...")
        try:
            blood_edges = load_network(omics_type, "blood")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue
        
        # 3. 加载Hub基因
        print("\n[3] 加载Hub基因...")
        hub_file = STEP3_DIR / omics_type / "brain_hubs.csv"
        if not hub_file.exists():
            print(f"  警告: Hub文件不存在 {hub_file}")
            print(f"  请先运行 step3_multixrank.py")
            continue
        
        hubs_df = pd.read_csv(hub_file)
        hub_genes = hubs_df['hub'].tolist()
        print(f"  加载 {len(hub_genes)} 个Hub基因")
        
        # 4. 识别脑端疾病相关基因（Gene Significance）
        print("\n[4] 识别脑端疾病相关基因（Gene Significance）...")
        brain_disease_genes, brain_gs_df = identify_disease_associated_genes(
            omics_type, tissue='brain', pvalue_threshold=0.05
        )
        
        # 5. 识别血端疾病相关基因（Gene Significance）
        print("\n[5] 识别血端疾病相关基因（Gene Significance）...")
        blood_disease_genes, blood_gs_df = identify_disease_associated_genes(
            omics_type, tissue='blood', pvalue_threshold=0.05
        )
        
        # 6. 脑端直接使用全部 Hub 基因（Hub 来自疾病模型 Jacobian 网络，拓扑中心性本身是疾病相关证据）
        print("\n[6] 脑端直接使用全部 Hub 基因（不做 GS overlap 筛选）...")
        brain_overlap = set(hub_genes)
        print(f"  Hub基因: {len(brain_overlap)} 个")
        
        # 7. 筛选跨组织边（overlap基因 → 疾病相关基因）
        print("\n[7] 筛选跨组织边（overlap基因 → 疾病相关基因）...")
        cross_tissue_file = STEP2_DIR / omics_type / "cross_tissue_edges.csv"
        if not cross_tissue_file.exists():
            print(f"  ✗ 找不到跨组织边文件: {cross_tissue_file}")
            continue
        
        cross_edges = pd.read_csv(cross_tissue_file)
        print(f"  原始跨组织边: {len(cross_edges)} 条")
        
        # 排除自环边（source == target）
        # 注意：蛋白质名转换延后到step4，避免BD-MAPT→MAPT后被当作自环边排除
        cross_edges = cross_edges[cross_edges['source'] != cross_edges['target']]
        print(f"  排除自环边后: {len(cross_edges)} 条")
        
        # 筛选：source 是 Hub 基因，target 是全部跨组织边对应的血端基因
        blood_disease_set = set(cross_edges['target'].unique())
        filtered_edges = cross_edges[
            cross_edges['source'].isin(brain_overlap) &
            cross_edges['target'].isin(blood_disease_set)
        ]
        
        print(f"  筛选后跨组织边: {len(filtered_edges)} 条")
        print(f"    脑overlap节点数: {filtered_edges['source'].nunique()}")
        print(f"    血疾病相关基因节点数: {filtered_edges['target'].nunique()}")
        
        # 创建输出目录（提前）
        omics_output = OUTPUT_DIR / omics_type
        omics_output.mkdir(exist_ok=True)
        
        # 8. 保存 overlap 边
        print(f"\n[8] 保存 overlap 边: {len(filtered_edges)} 条")
        filtered_edges.to_csv(omics_output / 'filtered_cross_tissue_edges.csv', index=False)
        print(f"  保存 filtered_cross_tissue_edges.csv: {len(filtered_edges)} 条")

        # 9. 与Hub基因比较
        print("\n" + "="*60)
        print("脑端疾病相关基因与Hub基因的交集:")
        print("="*60)
        brain_comparison = compare_with_hubs(brain_disease_genes, hub_genes)

        # 11. 保存结果
        omics_output = OUTPUT_DIR / omics_type
        omics_output.mkdir(exist_ok=True)
        
        # 9. 与Hub基因比较
        print("\n" + "="*60)
        print("脑端疾病相关基因与Hub基因的交集:")
        print("="*60)
        brain_comparison = compare_with_hubs(brain_disease_genes, hub_genes)
        
        # 11. 保存结果
        omics_output = OUTPUT_DIR / omics_type
        omics_output.mkdir(exist_ok=True)
        
        # 保存脑端疾病相关基因列表
        brain_gs_df.to_csv(omics_output / 'brain_disease_genes.csv', index=False)
        
        # 保存血端疾病相关基因列表
        blood_gs_df.to_csv(omics_output / 'blood_disease_genes.csv', index=False)
        
        # 保存脑overlap
        pd.DataFrame({
            'gene': list(brain_overlap),
            'type': 'brain_overlap_hub_disease'
        }).to_csv(omics_output / 'brain_overlap_hub_disease.csv', index=False)
        
        # 蛋白质组学：映射蛋白质名到基因symbol（在筛选后进行，避免真自环边被提前排除）
        # 注意：转换后产生的"自环边"（如BD-MAPT→MAPT）不是真自环边，有生物学意义，不应排除
        if omics_type == 'proteomics':
            print(f"\n[蛋白质名转换] 映射蛋白质名到基因symbol...")
            print(f"  转换前边数: {len(filtered_edges)}")
            filtered_edges['source'] = filtered_edges['source'].apply(map_protein_to_gene)
            filtered_edges['target'] = filtered_edges['target'].apply(map_protein_to_gene)
            
            # 合并重复边（使用Stouffer's method）
            print(f"  [蛋白质名转换] 合并重复边...")
            filtered_edges = merge_duplicate_edges_stouffer(filtered_edges)
            print(f"  最终边数: {len(filtered_edges)} 条")
        
        # 保存最终筛选后的跨组织边
        filtered_edges.to_csv(omics_output / 'filtered_cross_tissue_edges.csv', index=False)

        # 保存交集分析
        pd.DataFrame({
            'gene': brain_comparison['overlap'],
            'type': 'overlap_disease_hub'
        }).to_csv(omics_output / 'overlap_disease_hub.csv', index=False)
        
        # 保存统计摘要
        summary = {
            'omics': omics_type,
            'n_hub_genes': len(hub_genes),
            'brain': {
                'n_disease_genes': len(brain_disease_genes),
                'overlap_with_hubs': len(brain_comparison['overlap']),
                'overlap_ratio': brain_comparison['overlap_ratio'],
            },
            'blood': {
                'n_disease_genes': len(blood_disease_genes),
            },
            'brain_overlap': len(brain_overlap),
            'filtered_edges': {
                'n_edges': len(filtered_edges),
                'brain_nodes': int(filtered_edges['source'].nunique()),
                'blood_nodes': int(filtered_edges['target'].nunique()),
            }
        }
        
        import json
        with open(omics_output / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ 结果保存到: {omics_output}")
        

    print("\n" + "="*60)
    print("✅ 全部完成！")
    print("="*60)


if __name__ == '__main__':
    main()
