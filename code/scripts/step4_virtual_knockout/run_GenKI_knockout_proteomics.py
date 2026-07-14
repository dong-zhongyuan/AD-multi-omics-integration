#!/usr/bin/env python3
"""
GenKI Virtual Knockout Script (TMS adapter)
基于 GenKI 仓库的完整虚拟敲除实现（VGAE-based）

⚠️ 2026-07-11 更新：
  蛋白组不再走 VK（改用 STRING PPI 验证），只对转录组做 GenKI VK。
  代谢组改用 KEGG MRN 注释（见 step4_metabolomics_mrn.py）。

用法：
  # 正向敲除（默认）：敲除脑端基因，观察血液端基因
  python run_GenKI_knockout.py

  # 反向敲除：敲除血液端基因，观察脑端基因
  python run_GenKI_knockout.py --direction reverse

  # VK 只处理 transcriptomics（蛋白组改走 PPI）
  python run_GenKI_knockout.py --omics transcriptomics
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# 先导入配置管理器（在添加GenKI路径之前）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from tools.config_loader import get_config
from tools.gene_id_converter import ensembl_to_symbol
config = get_config()

# 路径转换函数
def rstr(path):
    """Convert Path to raw string"""
    return str(path).replace('\\', '/')

# 再添加 GenKI 到路径
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "GenKI-master"))

from GenKI.train import VGAE_trainer, VariationalGCNEncoder
from GenKI.model import VGAE
from GenKI.preprocesing import load_gdata, split_data
from GenKI.utils import get_distance, get_generank
from GenKI.pcNet import pcNet

# ============================================================================
# 配置参数（全局变量，将在main中根据命令行参数设置）
# ============================================================================

# 输入输出路径（用 PROJECT_ROOT 避免 /mnt/d/ 问题）
SC_DATA_PATH = str(PROJECT_ROOT / "processed-data" / "step4_single_cell_5xfad" / "5xFAD_expression_matrix_for_step4.h5ad")
STEP3_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "step3_hub_identification")
OUTPUT_DIR = None  # 将在main中设置
DIRECTION = "forward"  # 将在main中设置

# 组学列表（将在main中设置）
# 蛋白组和转录组都走 VK（PPI 无法体现跨组织方向性）
OMICS_LIST = ["proteomics", "transcriptomics"]

# VGAE 训练参数（严格按照 GenKI 默认值）
OUT_CHANNELS = int(config.get_parameter("virtual_knockout.genki.out_channels"))  # 潜在空间维度
EPOCHS = int(config.get_parameter("virtual_knockout.genki.epochs"))  # 训练轮数
LR = float(config.get_parameter("virtual_knockout.genki.learning_rate"))  # 学习率
WEIGHT_DECAY = float(config.get_parameter("virtual_knockout.genki.weight_decay"))  # 权重衰减
BETA = float(config.get_parameter("virtual_knockout.genki.beta"))  # KL 散度权重（beta-VAE）
RANDOM_STATE = int(config.get_parameter("virtual_knockout.genki.random_state"))  # 随机种子

# PCNet 参数
N_COMP = int(config.get_parameter("virtual_knockout.genki.n_comp"))  # PCA 成分数
SCALE_SCORES = True  # 是否标准化（GenKI默认）
SYMMETRIC = True  # 是否对称化（GenKI默认）
Q = 0.5  # 降低阈值从0.85→0.5，保留更多包含低表达基因的边

# 虚拟敲除参数
N_PERMUTATIONS = 10  # 置换检验次数（优化：从30降到10，节省67%时间）
PERMUTATION_EPOCHS = 10  # 置换检验的训练轮数（优化：从20降到10，节省50%时间）
DISTANCE_METHOD = "KL"  # 距离度量方法（KL 散度）

# ============================================================================
# 辅助函数
# ============================================================================


def compute_common_hvg(h5ad_path, tissue_types=["Brain", "Blood"], target_genes=None, n_top_genes=500):
    """
    在合并的多组织数据上计算HVG，确保所有组织使用相同的基因集
    
    Args:
        h5ad_path: h5ad 文件路径
        tissue_types: 组织类型列表
        target_genes: 必须保护的基因列表
        n_top_genes: HVG数量
    
    Returns:
        统一的基因列表（HVG + target_genes）
    """
    import scanpy as sc
    import warnings
    
    print(f"[Computing common HVG across tissues: {', '.join(tissue_types)}]")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        adata = sc.read_h5ad(h5ad_path, backed="r")
    
    # 1. 获取基因交集
    gene_sets = []
    for tissue in tissue_types:
        tissue_mask = adata.obs["tissue"] == tissue
        tissue_indices = np.where(tissue_mask)[0]
        if len(tissue_indices) > 0:
            genes = set(adata.var.index)
            gene_sets.append(genes)
            print(f"  {tissue}: {len(genes)} genes")
    
    # 取交集
    if len(gene_sets) == 0:
        return []
    
    common_genes = set.intersection(*gene_sets)
    print(f"  Common genes (intersection): {len(common_genes)}")
    
    # 2. 在交集基因上合并所有组织的数据
    print(f"  Loading merged data for HVG computation...")
    all_cells = []
    for tissue in tissue_types:
        tissue_mask = adata.obs["tissue"] == tissue
        tissue_indices = np.where(tissue_mask)[0]
        if len(tissue_indices) > 0:
            # 随机采样细胞（避免内存爆炸）
            if len(tissue_indices) > 3000:
                sampled_indices = np.random.choice(tissue_indices, 3000, replace=False)
            else:
                sampled_indices = tissue_indices
            all_cells.extend(sampled_indices)
    
    # 加载合并数据到内存
    adata_merged = adata[all_cells, :].to_memory()
    
    # 筛选到交集基因
    gene_mask = adata_merged.var.index.isin(common_genes)
    adata_merged = adata_merged[:, gene_mask]
    print(f"  Merged data: {adata_merged.shape[0]} cells × {adata_merged.shape[1]} genes")
    
    # 处理重复基因名
    adata_merged.var_names_make_unique()
    
    # 3. 计算HVG
    print(f"  Computing highly variable genes (n_top_genes={n_top_genes})...")
    import scanpy as sc
    sc.pp.highly_variable_genes(
        adata_merged,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        subset=False
    )
    
    hvg_genes = adata_merged.var.index[adata_merged.var['highly_variable']].tolist()
    print(f"  HVG selected: {len(hvg_genes)}")
    
    # 4. 保护target_genes
    if target_genes is not None:
        target_genes_unique = list(set(target_genes))
        available_targets = [g for g in target_genes_unique if g in common_genes]
        missing_targets = [g for g in target_genes_unique if g not in common_genes]
        
        print(f"  Target genes (unique): {len(target_genes_unique)}")
        print(f"  Available targets in common genes: {len(available_targets)}")
        
        if missing_targets:
            print(f"  WARNING: {len(missing_targets)} target genes not in common genes:")
            print(f"    {missing_targets[:10]}")
        
        # 合并HVG和target_genes
        final_genes = list(set(hvg_genes) | set(available_targets))
    else:
        final_genes = hvg_genes
    
    print(f"  Final gene set: {len(final_genes)} genes (HVG + targets)")
    return final_genes


def load_single_cell_data_h5ad(
    h5ad_path, tissue_type, target_genes=None, max_cells=3000, common_genes=None
):
    """
    从 h5ad 文件加载单细胞数据（流式处理，内存优化）

    Args:
        h5ad_path: h5ad 文件路径
        tissue_type: 组织类型过滤（'CSF' 或 'PBMC'）
        target_genes: 目标基因列表（可选，必须是交集基因的子集）
        max_cells: 最大细胞数（可选）
        common_genes: 交集基因列表（可选，如果提供则只在这些基因上选择HVG）

    Returns:
        genes x cells 的 DataFrame
    """
    # 动态降级策略：3000 → 2000 → 1000 → 500
    cell_limits = [max_cells, 2000, 1000, 500] if max_cells >= 3000 else [max_cells]
    
    for attempt, limit in enumerate(cell_limits):
        try:
            if attempt > 0:
                print(f"  Retrying with max_cells={limit} (attempt {attempt + 1}/{len(cell_limits)})...")
            return _load_single_cell_data_h5ad_impl(h5ad_path, tissue_type, target_genes, limit, common_genes)
        except (MemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  OOM with max_cells={limit}: {str(e)}")
                if attempt < len(cell_limits) - 1:
                    # 清理内存后重试
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  All attempts failed, giving up")
                    return None
            else:
                raise
    
    return None


def _load_single_cell_data_h5ad_impl(
    h5ad_path, tissue_type, target_genes=None, max_cells=3000, common_genes=None, max_genes=500
):
    """
    从 h5ad 文件加载单细胞数据（流式处理，内存优化）

    Args:
        h5ad_path: h5ad 文件路径
        tissue_type: 组织类型过滤（'CSF' 或 'PBMC'）
        target_genes: 目标基因列表（已弃用，使用common_genes代替）
        max_cells: 最大细胞数（可选）
        common_genes: 预先计算好的统一基因列表（如果提供，直接使用，不再计算HVG）
        max_genes: 最大基因数（仅在未提供common_genes时使用）

    Returns:
        genes x cells 的 DataFrame
    """
    import scanpy as sc
    import warnings

    print(f"[1/4] Loading h5ad file: {h5ad_path}")

    # 加载 h5ad 文件（先用 backed mode 读取元数据）
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        adata = sc.read_h5ad(h5ad_path, backed="r")

    print(f"  Loaded {adata.shape[0]} cells × {adata.shape[1]} genes")

    # 组织类型过滤
    print(f"[2/4] Filtering cells by tissue: {tissue_type}...")
    tissue_mask = adata.obs["tissue"] == tissue_type
    tissue_indices = np.where(tissue_mask)[0]

    if len(tissue_indices) == 0:
        print(f"  ERROR: No cells found for tissue {tissue_type}")
        return None

    print(f"  Found {len(tissue_indices)} {tissue_type} cells")

    # 限制细胞数
    if max_cells and len(tissue_indices) > max_cells:
        np.random.seed(42)
        tissue_indices = np.random.choice(tissue_indices, max_cells, replace=False)
        print(f"  Sampled to {max_cells} cells")

    # 基因选择：使用预先计算好的统一基因列表
    print(f"[3/4] Selecting genes...")
    
    if common_genes is not None:
        # 使用预先计算好的统一基因列表（已包含HVG + targets）
        print(f"  Using pre-computed common gene list: {len(common_genes)} genes")
        available_genes = [g for g in common_genes if g in adata.var.index]
        missing_genes = [g for g in common_genes if g not in adata.var.index]
        
        print(f"  Available genes in {tissue_type} data: {len(available_genes)}")
        if missing_genes:
            print(f"  WARNING: {len(missing_genes)} genes not found in {tissue_type} data")
            if len(missing_genes) <= 10:
                print(f"    Missing: {missing_genes}")
    else:
        # 旧逻辑：保护target_genes + 计算HVG（仅用于向后兼容）
        print(f"  WARNING: common_genes not provided, falling back to old logic")
        available_genes_in_data = list(adata.var.index)
        
        # 先保护 target_genes
        if target_genes is not None:
            target_genes_unique = list(set(target_genes))
            available_targets = [g for g in target_genes_unique if g in available_genes_in_data]
            missing_targets = [g for g in target_genes_unique if g not in available_genes_in_data]
            
            print(f"  Target genes (unique): {len(target_genes_unique)}")
            print(f"  Available targets: {len(available_targets)}")
            
            if missing_targets:
                print(f"  WARNING: {len(missing_targets)} target genes not found:")
                print(f"    {missing_targets[:10]}")
        else:
            available_targets = []
        
        # 如果需要更多基因，计算 HVG 补充
        if len(available_targets) < max_genes:
            # 先提取组织子集用于计算 HVG
            adata_subset = adata[tissue_indices, :].to_memory()
            
            # 处理重复基因名
            adata_subset.var_names_make_unique()
            
            # 计算 HVG（使用 Scanpy 标准流程）
            print(f"  Computing highly variable genes...")
            import scanpy as sc
            sc.pp.highly_variable_genes(
                adata_subset,
                n_top_genes=max_genes,
                flavor='seurat_v3',
                subset=False
            )
            
            hvg_genes = adata_subset.var_names[adata_subset.var['highly_variable']].tolist()
            print(f"  Found {len(hvg_genes)} HVGs")
            
            # 合并 target genes 和 HVG
            available_genes = list(set(available_targets + hvg_genes))
            print(f"  Final gene set: {len(available_genes)} genes ({len(available_targets)} targets + {len(hvg_genes)} HVGs)")
            
            # 清理临时对象
            del adata_subset
            gc.collect()
        else:
            available_genes = available_targets
            print(f"  Final gene set: {len(available_genes)} genes (all targets)")
    
    # 获取基因索引
    gene_indices = [adata.var.index.get_loc(g) for g in available_genes]

    # 流式读取数据（边读边写，不累积在内存）
    print(f"[4/4] Loading expression data (streaming)...")
    batch_size = 200
    n_batches = (len(tissue_indices) + batch_size - 1) // batch_size

    # 预分配数组（避免动态扩展）
    n_cells = len(tissue_indices)
    n_genes = len(available_genes)
    expr_array = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    from tqdm import tqdm
    for i in tqdm(range(n_batches), desc="  Loading batches", ncols=80):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_cells)
        batch_indices = tissue_indices[start_idx:end_idx]

        # 读取批次数据
        batch_data = adata[batch_indices, gene_indices].X
        if hasattr(batch_data, "toarray"):
            batch_data = batch_data.toarray()

        # 直接写入预分配的数组
        expr_array[start_idx:end_idx, :] = batch_data
        
        # 立即清理
        del batch_data
        gc.collect()

    # 创建 DataFrame
    cell_names = adata.obs.index[tissue_indices].tolist()
    expr_df = pd.DataFrame(
        expr_array.T,  # 转置为 genes x cells
        index=available_genes,
        columns=cell_names,
    )

    # 清理内存
    del expr_array, adata
    gc.collect()

    print(f"  Final shape: {expr_df.shape[0]} genes x {expr_df.shape[1]} cells")
    return expr_df


def aggregate_to_metacells(expr_df, n_metacells=100, min_cells_per_metacell=10):
    """
    将单细胞数据聚合为 metacells，提高低表达基因的检测率
    
    Args:
        expr_df: genes x cells DataFrame
        n_metacells: 目标 metacell 数量
        min_cells_per_metacell: 每个 metacell 最少细胞数
    
    Returns:
        genes x metacells DataFrame
    """
    print(f"  [Metacell] Aggregating {expr_df.shape[1]} cells into ~{n_metacells} metacells...")
    
    # 检查并去重基因索引
    if not expr_df.index.is_unique:
        print(f"    WARNING: Duplicate gene names detected, keeping first occurrence...")
        expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
    
    # 使用简单的 k-means 聚类（基于基因表达相似性）
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    
    # 转置为 cells x genes，标准化
    X = expr_df.T.values
    print(f"    Standardizing expression matrix...")
    scaler = StandardScaler(with_mean=False)  # sparse-friendly
    X_scaled = scaler.fit_transform(X)
    
    # K-means 聚类
    print(f"    Running MiniBatchKMeans clustering...")
    n_clusters = min(n_metacells, expr_df.shape[1] // min_cells_per_metacell)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(X_scaled)
    
    # 聚合每个 cluster 的细胞（取平均值）
    print(f"    Aggregating cells within each cluster...")
    metacell_list = []
    metacell_names = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_cells_in_cluster = cluster_mask.sum()
        
        if n_cells_in_cluster >= min_cells_per_metacell:
            # 取该 cluster 所有细胞的平均表达
            cluster_expr = expr_df.iloc[:, cluster_mask].mean(axis=1).values  # 转为 numpy array
            metacell_list.append(cluster_expr)
            metacell_names.append(f"metacell_{cluster_id}")
    
    # 创建 metacell DataFrame（使用 numpy array 避免索引冲突）
    metacell_df = pd.DataFrame(
        np.column_stack(metacell_list),
        index=expr_df.index,
        columns=metacell_names
    )
    
    print(f"    Created {len(metacell_names)} metacells (avg {expr_df.shape[1] / len(metacell_names):.1f} cells/metacell)")
    
    # 清理内存
    del X, X_scaled, scaler, kmeans
    gc.collect()
    
    return metacell_df


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


def load_cross_tissue_hubs_and_backtrace(omics_name, direction='forward'):
    """
    从 eigengene_analysis 加载筛选后的边，提取knockout目标基因
    
    Args:
        omics_name: 组学类型
        direction: 'forward' (敲除脑端) 或 'reverse' (敲除血液端)
    
    返回: [(gene, category), ...] 列表, 以及所有target基因列表
    """
    # 从 eigengene_analysis 加载筛选后的边
    # 转录组用预筛选文件，其他用原始 filtered 文件
    if omics_name == "transcriptomics":
        prescreened = Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / omics_name / "prescreened_cross_tissue_edges.csv"
        if prescreened.exists():
            edges_file = prescreened
            print(f"  Using prescreened edges: {edges_file}")
        else:
            edges_file = Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / omics_name / "filtered_cross_tissue_edges.csv"
    else:
        edges_file = Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / omics_name / "filtered_cross_tissue_edges.csv"

    if not edges_file.exists():
        print(f"  WARNING: {edges_file} not found")
        return [], []

    edges_df = pd.read_csv(edges_file)
    print(f"  Loaded {len(edges_df)} filtered edges from {edges_file}")
    
    # 提取唯一的脑端基因（source列）和血液端基因（target列）
    brain_genes = edges_df['source'].unique().tolist()
    blood_genes = edges_df['target'].unique().tolist()
    
    # 根据方向选择knockout目标和observation目标
    if direction == 'reverse':
        # 反向：敲除血液端基因，观察脑端基因
        knockout_genes = blood_genes
        observation_genes = brain_genes
        print(f"  Extracted {len(knockout_genes)} unique blood genes (knockout targets - REVERSE)")
        print(f"  Extracted {len(observation_genes)} unique brain genes (observation targets - REVERSE)")
    else:
        # 正向（默认）：敲除脑端基因，观察血液端基因
        knockout_genes = brain_genes
        observation_genes = blood_genes
        print(f"  Extracted {len(knockout_genes)} unique brain genes (knockout targets - FORWARD)")
        print(f"  Extracted {len(observation_genes)} unique blood genes (observation targets - FORWARD)")
    
    # 所有基因标记为 'eigengene_filtered' 类别
    hub_genes_with_category = [(gene, 'eigengene_filtered') for gene in knockout_genes]
    
    # 转录组学：5xFAD 数据已经是 gene symbol（人同源），不需要 Ensembl 转换
    if omics_name == "transcriptomics":
        print(f"  Transcriptomics genes already in symbol format (5xFAD ortholog-mapped), skipping Ensembl conversion", flush=True)
        print(f"  Converting Ensembl IDs to gene symbols...", flush=True)

        # 收集所有需要转换的Ensembl ID
        ensembl_ids = [gene_id for gene_id, _ in hub_genes_with_category]
        
        # 使用统一的ID转换工具
        id_mapping = ensembl_to_symbol(ensembl_ids)
        print(f"  Loaded {len(id_mapping)} Ensembl ID -> Gene Symbol mappings", flush=True)

        # 转换基因名
        converted = []
        not_found_unique = set()
        for gene_id, category in hub_genes_with_category:
            if gene_id in id_mapping:
                gene_symbol = id_mapping[gene_id]
                converted.append((gene_symbol, category))
            else:
                # 如果找不到，记录（去重）
                not_found_unique.add(gene_id)

        if not_found_unique:
            print(f"  WARNING: Could not convert {len(not_found_unique)} unique Ensembl IDs: {', '.join(sorted(list(not_found_unique)[:10]))}", flush=True)

        hub_genes_with_category = converted
        print(f"  Successfully converted {len(converted)} genes to symbols", flush=True)

    # 蛋白质组学：source列是蛋白质名，需要映射到基因symbol
    if omics_name == "proteomics":
        print(f"  Mapping protein names to gene symbols...", flush=True)
        
        converted = []
        mapping_log = {}
        for protein_name, category in hub_genes_with_category:
            gene_symbol = map_protein_to_gene(protein_name)
            converted.append((gene_symbol, category))
            
            # 记录映射（用于调试）
            if protein_name != gene_symbol:
                mapping_log[protein_name] = gene_symbol
        
        if mapping_log:
            print(f"  Mapped {len(mapping_log)} protein names:", flush=True)
            for protein, gene in sorted(mapping_log.items())[:10]:
                print(f"    {protein} -> {gene}", flush=True)
            if len(mapping_log) > 10:
                print(f"    ... and {len(mapping_log) - 10} more", flush=True)
        
        hub_genes_with_category = converted
        print(f"  Successfully mapped {len(converted)} proteins to gene symbols", flush=True)

    # 对observation_genes进行相同的映射处理
    observation_genes_mapped = []
    if omics_name == "transcriptomics":
        # 5xFAD 数据已经是 gene symbol，不需要转换
        observation_genes_mapped = observation_genes
        print(f"  Observation genes already in symbol format, using directly", flush=True)
    elif omics_name == "proteomics":
        print(f"  Mapping {len(observation_genes)} observation gene names to symbols...", flush=True)
        for gene_name in observation_genes:
            gene_symbol = map_protein_to_gene(gene_name)
            observation_genes_mapped.append(gene_symbol)
        print(f"  Successfully mapped {len(observation_genes_mapped)} observation genes to symbols", flush=True)
    else:
        observation_genes_mapped = observation_genes
    
    # 统计去重后的基因数
    unique_genes = list(set([gene for gene, _ in hub_genes_with_category]))
    unique_observation_genes = list(set(observation_genes_mapped))
    print(f"  Total hub genes: {len(hub_genes_with_category)} (with category duplicates)")
    print(f"  Unique hub genes: {len(unique_genes)}")
    print(f"  Unique observation genes: {len(unique_observation_genes)}")
    return hub_genes_with_category, observation_genes_mapped


def build_pcnet(data_df, n_comp=3, scale_scores=True, symmetric=False, q=0.95):
    """
    构建 PC 网络（Principal Component Network）
    严格遵循 GenKI 的 pcNet 实现，批量处理以节省内存（更激进）
    """
    print(f"  Building PCNet...")
    print(f"    Parameters: n_comp={n_comp}, q={q}")

    # 转置为 cells x genes
    X = data_df.T.values

    # 填充NaN（用0替代）
    if np.isnan(X).any():
        print(f"    WARNING: Found {np.isnan(X).sum()} NaN values, filling with 0...")
        X = np.nan_to_num(X, nan=0.0)

    # Quantile normalization（让低表达基因的方差不被高表达基因压制）
    print(f"    Applying quantile normalization...")
    for j in range(X.shape[1]):
        col = X[:, j]
        order = np.argsort(col)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(col))
        sorted_vals = np.sort(col)
        X[:, j] = sorted_vals[ranks]

    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # 避免除以0：如果std为0，设为1（这些基因不会有贡献）
    X_std[X_std < 1e-8] = 1.0

    X = (X - X_mean) / X_std
    
    # 再次检查NaN（防御性编程）
    if np.isnan(X).any():
        print(f"    WARNING: NaN after standardization, filling with 0...")
        X = np.nan_to_num(X, nan=0.0)

    # 构建邻接矩阵（批量处理）
    n_genes = X.shape[1]
    A = np.zeros((n_genes, n_genes), dtype=np.float32)

    # 对每个基因计算 PC 回归系数（分批处理，优化batch size）
    from sklearn.decomposition import PCA

    batch_size = 100  # 优化：提高到100，减少循环次数
    n_batches = (n_genes + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_genes)

        for k in range(start_idx, end_idx):
            y = X[:, k]
            Xi = np.delete(X, k, axis=1)  # 移除当前基因

            # PCA 降维
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            
            try:
                scores = pca.fit_transform(Xi)
            except np.linalg.LinAlgError as e:
                # SVD不收敛：使用更鲁棒的方法
                print(f"    WARNING: SVD failed for gene {k}, using randomized PCA...")
                pca = PCA(n_components=n_comp, random_state=RANDOM_STATE, svd_solver='randomized')
                try:
                    scores = pca.fit_transform(Xi)
                except Exception as e2:
                    # 如果还是失败，跳过这个基因（设置为0）
                    print(f"    WARNING: Randomized PCA also failed for gene {k}, skipping...")
                    continue

            # 标准化 scores
            scores = scores / np.sqrt(np.sum(scores**2, axis=0, keepdims=True))

            # 计算回归系数
            coef = pca.components_.T  # (genes-1) x n_comp
            betas = coef @ (scores.T @ y)

            # 填充邻接矩阵
            A[k, :k] = betas[:k]
            A[k, k + 1 :] = betas[k:]

            # 清理
            del y, Xi, pca, scores, coef, betas

        # 批次完成后强制清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:  # 优化：降低打印频率
            print(f"    Processed {end_idx}/{n_genes} genes")

    # 标准化和阈值化
    if scale_scores:
        A = A / np.max(np.abs(A))

    A[np.abs(A) < np.quantile(np.abs(A), q)] = 0

    if symmetric:
        A = (A + A.T) / 2

    print(f"    PCNet shape: {A.shape}, non-zero edges: {np.count_nonzero(A)}")

    # 清理内存
    del X
    gc.collect()

    return A


def create_pyg_data(data_df, adj_matrix):
    """
    创建 PyTorch Geometric Data 对象（内存优化）
    """
    print(f"  Creating PyG Data object...")

    # 节点特征：基因表达矩阵（genes x cells）
    x = torch.tensor(data_df.values, dtype=torch.float32)

    # 边索引：从邻接矩阵提取
    adj_sparse = csr_matrix(adj_matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)

    # 清理邻接矩阵
    del adj_matrix, adj_sparse
    gc.collect()

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight.float())

    print(f"    Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    return data


def _single_permutation_worker(args):
    """
    单次置换检验的工作函数（用于并行处理）
    
    Args:
        args: tuple of (perm_i, csf_data_df, pbmc_data_df, gene_to_ko, n_pbmc_cells, seed)
    
    Returns:
        float: 该次置换的平均KL散度
    """
    perm_i, csf_data_df, pbmc_data_df, gene_to_ko, n_pbmc_cells, seed = args
    
    try:
        # 设置随机种子
        np.random.seed(seed + perm_i)
        
        # 生成随机置换索引（只用于 PBMC 细胞列）
        perm_indices = np.random.permutation(n_pbmc_cells)
        
        # 置换 PBMC 细胞列（CSF 保持不变）
        wt_pbmc_perm = pbmc_data_df.iloc[:, perm_indices]
        ko_pbmc_perm = pbmc_data_df.iloc[:, perm_indices]
        
        # 确保基因顺序一致（取交集）
        common_genes = csf_data_df.index.intersection(wt_pbmc_perm.index)
        csf_data_perm = csf_data_df.loc[common_genes]
        wt_pbmc_perm = wt_pbmc_perm.loc[common_genes]
        ko_pbmc_perm = ko_pbmc_perm.loc[common_genes]
        
        # 准备 KO 的 CSF 数据
        ko_csf_data_df_perm = csf_data_perm.copy()
        ko_csf_data_df_perm.loc[gene_to_ko, :] = 0
        
        # 重新构建数据（CSF 固定 + PBMC 置换）
        wt_data_perm = pd.concat([csf_data_perm, wt_pbmc_perm], axis=1)
        ko_data_perm = pd.concat([ko_csf_data_df_perm, ko_pbmc_perm], axis=1)
        
        # 立即清理中间变量
        del wt_pbmc_perm, ko_pbmc_perm, ko_csf_data_df_perm
        gc.collect()
        
        # 重新构建 PCNet 和训练 VGAE（WT）
        wt_adj_perm = build_pcnet(
            wt_data_perm, n_comp=N_COMP, scale_scores=SCALE_SCORES, symmetric=SYMMETRIC, q=Q
        )
        wt_data_pyg_perm = create_pyg_data(wt_data_perm, wt_adj_perm)
        
        # 清理WT数据和邻接矩阵
        del wt_data_perm, wt_adj_perm
        gc.collect()
        
        wt_trainer_perm = VGAE_trainer(
            data=wt_data_pyg_perm,
            out_channels=OUT_CHANNELS,
            epochs=PERMUTATION_EPOCHS,  # 优化：置换检验用更少epochs
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            beta=BETA,
            log_dir=None,
            verbose=False,
            seed=seed + perm_i,
        )
        wt_trainer_perm.train()
        z_mu_wt_perm, z_std_wt_perm = wt_trainer_perm.get_latent_vars(wt_data_pyg_perm)
        
        # 清理WT trainer和data
        del wt_data_pyg_perm, wt_trainer_perm
        gc.collect()
        
        # 重新构建 PCNet 和训练 VGAE（KO）
        ko_adj_perm = build_pcnet(
            ko_data_perm, n_comp=N_COMP, scale_scores=SCALE_SCORES, symmetric=SYMMETRIC, q=Q
        )
        ko_data_pyg_perm = create_pyg_data(ko_data_perm, ko_adj_perm)
        
        # 清理KO数据和邻接矩阵
        del ko_data_perm, ko_adj_perm
        gc.collect()
        
        ko_trainer_perm = VGAE_trainer(
            data=ko_data_pyg_perm,
            out_channels=OUT_CHANNELS,
            epochs=PERMUTATION_EPOCHS,  # 优化：置换检验用更少epochs
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            beta=BETA,
            log_dir=None,
            verbose=False,
            seed=seed + perm_i + 1000,
        )
        ko_trainer_perm.train()
        z_mu_ko_perm, z_std_ko_perm = ko_trainer_perm.get_latent_vars(ko_data_pyg_perm)
        
        # 清理KO trainer和data
        del ko_data_pyg_perm, ko_trainer_perm
        gc.collect()
        
        # 计算 KL 散度
        gene_kl_perm = get_distance(
            z_mu_ko_perm, z_std_ko_perm, z_mu_wt_perm, z_std_wt_perm, by=DISTANCE_METHOD
        )
        kl_mean = gene_kl_perm.mean()
        
        # 清理所有剩余变量
        del z_mu_wt_perm, z_std_wt_perm, z_mu_ko_perm, z_std_ko_perm, gene_kl_perm
        gc.collect()
        
        return kl_mean
        
    except Exception as e:
        print(f"    WARNING: Permutation {perm_i} failed: {str(e)}")
        return None


def virtual_knockout_genki_cross_tissue(
    csf_data_df, pbmc_data_df, gene_to_ko, output_prefix,
    z_mu_wt_shared=None, z_std_wt_shared=None, all_genes_list=None,
    n_csf_cells=None, n_pbmc_cells=None
):
    """
    执行 GenKI 跨组织虚拟敲除（并行化版本，优化：复用WT网络）

    核心逻辑（不可更改）：
    1. 合并 CSF 和 PBMC 数据
    2. 在 CSF 细胞中敲除基因（设置表达为 0）
    3. 在 PBMC 细胞中保持原样
    4. 只分析 PBMC 细胞的潜在空间变化

    Args:
        csf_data_df: CSF genes x cells 表达矩阵
        pbmc_data_df: PBMC genes x cells 表达矩阵
        gene_to_ko: 要敲除的基因名（在 CSF 中敲除）
        output_prefix: 输出文件前缀
        z_mu_wt_shared: 共享的WT潜在变量均值（优化：避免重复计算）
        z_std_wt_shared: 共享的WT潜在变量标准差（优化：避免重复计算）
        all_genes_list: 所有基因列表（优化：避免重复计算）
        n_csf_cells: CSF细胞数（优化：避免重复计算）
        n_pbmc_cells: PBMC细胞数（优化：避免重复计算）

    Returns:
        dict: {
            'gene_ranking': PBMC 基因响应排序,
            'statistics': 统计结果,
            'kl_divergence': KL 散度,
            'p_value': p 值
        }
    """
    is_reverse = DIRECTION == 'reverse'
    ko_tissue = 'PBMC' if is_reverse else 'CSF'
    obs_tissue = 'CSF' if is_reverse else 'PBMC'

    print(f"\n{'=' * 80}")
    print(f"GenKI Cross-Tissue Virtual Knockout: {gene_to_ko}")
    print(f"  Strategy: KO in {ko_tissue} -> Observe in {obs_tissue}")
    print(f"{'=' * 80}")

    # 检查基因是否存在（保护机制）
    if gene_to_ko not in csf_data_df.index:
        print(f"  ERROR: Gene {gene_to_ko} not found in CSF data")
        print(f"  Available genes in CSF: {csf_data_df.shape[0]}")
        return None
    if gene_to_ko not in pbmc_data_df.index:
        print(f"  ERROR: Gene {gene_to_ko} not found in PBMC data")
        print(f"  Available genes in PBMC: {pbmc_data_df.shape[0]}")
        return None

    # 确保基因顺序一致
    common_genes = csf_data_df.index.intersection(pbmc_data_df.index)
    csf_data_df = csf_data_df.loc[common_genes]
    pbmc_data_df = pbmc_data_df.loc[common_genes]

    print(f"  Common genes: {len(common_genes)}")
    print(f"  CSF cells: {csf_data_df.shape[1]}, PBMC cells: {pbmc_data_df.shape[1]}")

    # 验证目标基因仍在common_genes中
    if gene_to_ko not in common_genes:
        print(f"  ERROR: Gene {gene_to_ko} lost after intersection")
        return None

    # 调试：检查输入数据是否有重复索引
    print(
        f"  DEBUG: CSF unique genes: {len(csf_data_df.index.unique())}/{len(csf_data_df.index)}"
    )
    print(
        f"  DEBUG: PBMC unique genes: {len(pbmc_data_df.index.unique())}/{len(pbmc_data_df.index)}"
    )

    # ========================================================================
    # Step 1: 使用共享的WT网络（优化：跳过重复计算）
    # ========================================================================
    if z_mu_wt_shared is not None and z_std_wt_shared is not None:
        print(f"\n[Step 1/6] Using shared WT network (optimization enabled)...")
        z_mu_wt = z_mu_wt_shared
        z_std_wt = z_std_wt_shared
        all_genes = all_genes_list
        print(f"  WT latent variables: mu shape={z_mu_wt.shape}, std shape={z_std_wt.shape}")
        print(f"  Cell mask: {n_csf_cells} CSF + {n_pbmc_cells} PBMC")
    else:
        # 回退到原始逻辑（如果没有提供共享变量）
        print(f"\n[Step 1/6] Merging CSF and PBMC data (WT)...")

        wt_data_df = pd.concat([csf_data_df, pbmc_data_df], axis=1)
        n_csf_cells = csf_data_df.shape[1]
        n_pbmc_cells = pbmc_data_df.shape[1]

        # 保存所有基因列表（用于后续排序）
        all_genes = wt_data_df.index.tolist()

        # 调试：检查是否有重复
        print(f"  DEBUG: Total genes in all_genes: {len(all_genes)}")
        print(f"  DEBUG: Unique genes in all_genes: {len(set(all_genes))}")
        if len(all_genes) != len(set(all_genes)):
            print(f"  WARNING: Duplicate genes detected in all_genes!")
            from collections import Counter

            duplicates = [gene for gene, count in Counter(all_genes).items() if count > 1]
            print(f"  Duplicate genes: {duplicates[:10]}")

        print(f"  WT data shape: {wt_data_df.shape}")
        print(f"  Cell mask: {n_csf_cells} CSF + {n_pbmc_cells} PBMC")

        # 构建 WT PCNet 和训练 VGAE
        wt_adj = build_pcnet(
            wt_data_df, n_comp=N_COMP, scale_scores=SCALE_SCORES, symmetric=SYMMETRIC, q=Q
        )
        wt_data = create_pyg_data(wt_data_df, wt_adj)

        # 清理 wt_adj
        del wt_adj
        gc.collect()

        print(f"  Training WT VGAE...", flush=True)
        wt_trainer = VGAE_trainer(
            data=wt_data,
            out_channels=OUT_CHANNELS,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            beta=BETA,
            log_dir=None,
            verbose=True,  # 优化：主实验显示详细进度
            seed=RANDOM_STATE,
        )

        try:
            wt_trainer.train()
        except ValueError as e:
            if "Insufficient number of edges" in str(e):
                print(f"  ERROR: Insufficient edges for WT network training")
                print(f"  This gene has too sparse expression, skipping...")
                return None
            else:
                raise

        print(f"  WT VGAE training completed")
        epoch, loss, auc, ap = wt_trainer.final_metrics
        print(f"    Final metrics: Loss={loss:.4f}, AUROC={auc:.4f}, AP={ap:.4f}")

        # 获取 WT 潜在变量
        z_mu_wt, z_std_wt = wt_trainer.get_latent_vars(wt_data)
        print(
            f"  WT latent variables: mu shape={z_mu_wt.shape}, std shape={z_std_wt.shape}"
        )

        # 清理 wt_data_df（不再需要）
        del wt_data_df, wt_trainer, wt_data
        gc.collect()

    # ========================================================================
    # Step 2: 执行跨组织虚拟敲除
    # ========================================================================
    print(f"\n[Step 2/6] Performing cross-tissue virtual knockout...")

    # 复制数据（只复制需要敲除的一侧）
    if is_reverse:
        ko_csf_data_df = csf_data_df
        ko_pbmc_data_df = pbmc_data_df.copy()
        ko_pbmc_data_df.loc[gene_to_ko, :] = 0
        pbmc_wt_mean = pbmc_data_df.loc[gene_to_ko, :].values.mean()
        pbmc_ko_mean = ko_pbmc_data_df.loc[gene_to_ko, :].values.mean()
        csf_mean = csf_data_df.loc[gene_to_ko, :].values.mean()
        print(f"  Knocked out gene: {gene_to_ko} (PBMC only)")
        print(f"  PBMC WT expression (mean): {pbmc_wt_mean:.4f}")
        print(f"  PBMC KO expression (mean): {pbmc_ko_mean:.4f}")
        print(f"  CSF expression (unchanged): {csf_mean:.4f}")
    else:
        ko_csf_data_df = csf_data_df.copy()
        ko_pbmc_data_df = pbmc_data_df
        ko_csf_data_df.loc[gene_to_ko, :] = 0
        csf_wt_mean = csf_data_df.loc[gene_to_ko, :].values.mean()
        csf_ko_mean = ko_csf_data_df.loc[gene_to_ko, :].values.mean()
        pbmc_mean = pbmc_data_df.loc[gene_to_ko, :].values.mean()
        print(f"  Knocked out gene: {gene_to_ko} (CSF only)")
        print(f"  CSF WT expression (mean): {csf_wt_mean:.4f}")
        print(f"  CSF KO expression (mean): {csf_ko_mean:.4f}")
        print(f"  PBMC expression (unchanged): {pbmc_mean:.4f}")

    # 合并 KO 数据
    ko_data_df = pd.concat([ko_csf_data_df, ko_pbmc_data_df], axis=1)

    # 清理 ko_csf_data_df
    del ko_csf_data_df
    gc.collect()

    print(f"  KO data shape: {ko_data_df.shape}")

    # ========================================================================
    # Step 3: 构建 KO PCNet 和训练 VGAE
    # ========================================================================
    print(f"\n[Step 3/6] Building KO PCNet and training VGAE...")

    ko_adj = build_pcnet(
        ko_data_df, n_comp=N_COMP, scale_scores=SCALE_SCORES, symmetric=SYMMETRIC, q=Q
    )
    ko_data = create_pyg_data(ko_data_df, ko_adj)

    # 清理 ko_adj 和 ko_data_df
    del ko_adj, ko_data_df
    gc.collect()

    print(f"  Training KO VGAE...", flush=True)
    ko_trainer = VGAE_trainer(
        data=ko_data,
        out_channels=OUT_CHANNELS,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        beta=BETA,
        log_dir=None,
        verbose=True,  # 优化：主实验显示详细进度
        seed=RANDOM_STATE,
    )

    try:
        ko_trainer.train()
    except ValueError as e:
        if "Insufficient number of edges" in str(e):
            print(f"  ERROR: Insufficient edges for KO network training")
            print(f"  This gene has too sparse expression, skipping...")
            return None
        else:
            raise

    print(f"  KO VGAE training completed")
    epoch, loss, auc, ap = ko_trainer.final_metrics
    print(f"    Final metrics: Loss={loss:.4f}, AUROC={auc:.4f}, AP={ap:.4f}")

    # 获取 KO 潜在变量
    z_mu_ko, z_std_ko = ko_trainer.get_latent_vars(ko_data)
    print(
        f"  KO latent variables: mu shape={z_mu_ko.shape}, std shape={z_std_ko.shape}"
    )

    # ========================================================================
    # Step 4: 计算基因网络的 KL 散度
    # ========================================================================
    print(f"\n[Step 4/6] Computing KL divergence...", flush=True)
    print(f"  Note: VGAE outputs gene-level latent representations, not cell-level", flush=True)
    print(f"  Latent variables shape: {z_mu_wt.shape} (genes × latent_dim)", flush=True)
    
    # 诊断：检查潜在变量的数值范围
    print(f"  [Diagnostic] WT z_mu: min={z_mu_wt.min():.6f}, max={z_mu_wt.max():.6f}, mean={z_mu_wt.mean():.6f}", flush=True)
    print(f"  [Diagnostic] WT z_std: min={z_std_wt.min():.6f}, max={z_std_wt.max():.6f}, mean={z_std_wt.mean():.6f}", flush=True)
    print(f"  [Diagnostic] KO z_mu: min={z_mu_ko.min():.6f}, max={z_mu_ko.max():.6f}, mean={z_mu_ko.mean():.6f}", flush=True)
    print(f"  [Diagnostic] KO z_std: min={z_std_ko.min():.6f}, max={z_std_ko.max():.6f}, mean={z_std_ko.mean():.6f}", flush=True)

    # 计算每个基因的 KL 散度
    # get_distance() 遍历每个基因，计算其在 WT 和 KO 网络中的分布差异
    gene_kl_divergences = get_distance(
        z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by=DISTANCE_METHOD
    )

    # 使用平均 KL 散度作为整体度量
    dis_overall = gene_kl_divergences.mean()

    print(
        f"  Gene-wise KL divergences: min={gene_kl_divergences.min():.6f}, max={gene_kl_divergences.max():.6f}, mean={dis_overall:.6f}",
        flush=True
    )
    print(f"  Overall network KL divergence: {dis_overall:.6f}", flush=True)

    # 清理完整的潜在变量
    del z_mu_wt, z_std_wt, z_mu_ko, z_std_ko
    gc.collect()

    # ========================================================================
    # Step 5: 阴性对照分析（替代置换检验）
    # ========================================================================
    print(f"\n[Step 5/6] Negative control analysis...", flush=True)
    
    # 加载边文件，获取所有在因果网络中的基因
    if output_prefix == 'transcriptomics':
        prescreened = Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / output_prefix / "prescreened_cross_tissue_edges.csv"
        edges_file = prescreened if prescreened.exists() else Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / output_prefix / "filtered_cross_tissue_edges.csv"
    else:
        edges_file = Path(STEP3_OUTPUT_DIR) / "eigengene_analysis" / output_prefix / "filtered_cross_tissue_edges.csv"
    if edges_file.exists():
        edges_df = pd.read_csv(edges_file)

        # transcriptomics 边文件仍是 Ensembl ID，step4 当前在 symbol 空间里运行
        if output_prefix == 'transcriptomics':
            all_edge_ids = list(set(edges_df['source'].astype(str)).union(set(edges_df['target'].astype(str))))
            edge_mapping = ensembl_to_symbol(all_edge_ids)
            edges_df = edges_df.copy()
            edges_df['source'] = edges_df['source'].astype(str).map(lambda x: edge_mapping.get(x, x))
            edges_df['target'] = edges_df['target'].astype(str).map(lambda x: edge_mapping.get(x, x))

        edge_genes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
        print(f"  Loaded {len(edges_df)} edges, {len(edge_genes)} unique genes in causal network")
    else:
        print(f"  WARNING: Edge file not found: {edges_file}")
        edge_genes = set()
    
    # 计算每个基因的平均表达量（用于匹配阴性对照）
    print(f"  Computing gene expression levels...")
    gene_expressions = {}
    for gene in all_genes:
        if gene in csf_data_df.index and gene in pbmc_data_df.index:
            csf_expr = csf_data_df.loc[gene, :].values
            pbmc_expr = pbmc_data_df.loc[gene, :].values
            all_expr = np.concatenate([csf_expr, pbmc_expr])
            gene_expressions[gene] = np.mean(all_expr)
    
    print(f"  Computed expression for {len(gene_expressions)} genes")
    
    # 清理数据
    del csf_data_df, pbmc_data_df
    gc.collect()
    
    # 清理 trainers 和 data（如果存在）
    if 'wt_trainer' in locals():
        del wt_trainer
    if 'ko_trainer' in locals():
        del ko_trainer
    if 'wt_data' in locals():
        del wt_data
    if 'ko_data' in locals():
        del ko_data
    gc.collect()

    # 清理 PyTorch 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================================
    # Step 6: 基因排序 + 阴性对照统计检验
    # ========================================================================
    print(f"\n[Step 6/6] Ranking affected genes with negative control analysis...")

    # 调试：确保长度匹配
    print(f"  DEBUG: all_genes length: {len(all_genes)}")
    print(f"  DEBUG: gene_kl_divergences length: {len(gene_kl_divergences)}")
    
    if len(all_genes) != len(gene_kl_divergences):
        print(f"  ERROR: Length mismatch! all_genes={len(all_genes)}, gene_kl_divergences={len(gene_kl_divergences)}")
        return None

    # 使用基因的 KL 散度进行排序
    gene_rank_df = pd.DataFrame(
        {
            "Gene": all_genes,
            "KL_Divergence": gene_kl_divergences,
            "Z_score": (gene_kl_divergences - gene_kl_divergences.mean())
            / gene_kl_divergences.std(),
        }
    )

    gene_rank_df = gene_rank_df.sort_values("KL_Divergence", ascending=False)

    print(f"  Top 10 affected genes:")
    print(gene_rank_df.head(10))
    
    # 阴性对照分析：对每个在边文件中的target基因进行统计检验
    print(f"\n  Performing negative control analysis for target genes...")
    
    # 获取当前敲除基因的下游/上游 target 基因（依方向而定）
    if edges_file.exists():
        if is_reverse:
            ko_edges = edges_df[edges_df['target'] == gene_to_ko]
            target_genes = ko_edges['source'].unique().tolist()
        else:
            ko_edges = edges_df[edges_df['source'] == gene_to_ko]
            target_genes = ko_edges['target'].unique().tolist()
        print(f"  Found {len(target_genes)} target genes for {gene_to_ko}: {target_genes}")
    else:
        target_genes = []
    
    # 对每个target基因进行阴性对照分析
    negative_control_results = []
    N_NEGATIVE_CONTROLS = 10
    EXPRESSION_TOLERANCE = 0.3
    
    for target_gene in target_genes:
        if target_gene not in gene_rank_df['Gene'].values:
            print(f"    WARNING: {target_gene} not in gene_ranking, skipping...")
            continue
        
        # 获取target基因的KL散度
        target_kl = gene_rank_df[gene_rank_df['Gene'] == target_gene]['KL_Divergence'].values[0]
        target_rank = gene_rank_df[gene_rank_df['Gene'] == target_gene].index[0] + 1
        target_expr = gene_expressions.get(target_gene, None)
        
        if target_expr is None:
            print(f"    WARNING: Cannot get expression for {target_gene}")
            continue
        
        # 找表达量匹配的阴性对照
        expr_min = target_expr * (1 - EXPRESSION_TOLERANCE)
        expr_max = target_expr * (1 + EXPRESSION_TOLERANCE)
        
        candidates = []
        for gene in all_genes:
            if gene in edge_genes or gene == target_gene:
                continue
            gene_expr = gene_expressions.get(gene, None)
            if gene_expr is None:
                continue
            if expr_min <= gene_expr <= expr_max:
                gene_kl = gene_rank_df[gene_rank_df['Gene'] == gene]['KL_Divergence'].values[0]
                candidates.append((gene, gene_expr, gene_kl, abs(gene_expr - target_expr)))
        
        # 按表达量差异排序，选择最接近的N个
        candidates.sort(key=lambda x: x[3])
        negative_controls = candidates[:N_NEGATIVE_CONTROLS]
        
        if len(negative_controls) == 0:
            print(f"    WARNING: No negative controls found for {target_gene}")
            continue
        
        control_kls = np.array([kl for _, _, kl, _ in negative_controls])
        control_exprs = np.array([expr for _, expr, _, _ in negative_controls])
        control_genes = [gene for gene, _, _, _ in negative_controls]
        
        # 单样本t检验 (单侧检验：target > control_mean)
        from scipy import stats
        # 使用control分布检验target是否显著偏离
        t_statistic, p_value = stats.ttest_1samp(control_kls, target_kl, alternative='less')
        # alternative='less' 表示检验 control_mean < target_kl
        
        # Cohen's d
        control_mean = control_kls.mean()
        control_std = control_kls.std()
        cohens_d = (target_kl - control_mean) / control_std if control_std > 0 else np.nan
        
        print(f"    {target_gene}: KL={target_kl:.4f}, Controls mean={control_mean:.4f}, p={p_value:.4f}, d={cohens_d:.2f}")
        
        negative_control_results.append({
            'target_gene': target_gene,
            'target_kl': target_kl,
            'target_rank': target_rank,
            'target_expression': target_expr,
            'n_controls': len(negative_controls),
            'control_kl_mean': control_mean,
            'control_kl_std': control_std,
            'control_expression_mean': control_exprs.mean(),
            'control_expression_std': control_exprs.std(),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'negative_controls': ','.join(control_genes)
        })
    
    # 汇总统计
    if len(negative_control_results) > 0:
        n_significant = sum([r['significant'] for r in negative_control_results])
        print(f"\n  Negative control summary:")
        print(f"    Total target genes analyzed: {len(negative_control_results)}")
        print(f"    Significant (p<0.05): {n_significant}")
        print(f"    Significant rate: {n_significant/len(negative_control_results)*100:.1f}%")

    # ========================================================================
    # 保存结果（流式保存）
    # ========================================================================
    print(f"\n[Saving Results]")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存基因排序（去除重复基因，保留第一次出现的记录）
    rank_path = os.path.join(
        OUTPUT_DIR, f"{output_prefix}_{gene_to_ko}_gene_ranking.csv"
    )
    gene_rank_df_unique = gene_rank_df.drop_duplicates(subset=["Gene"], keep="first")
    gene_rank_df_unique.to_csv(rank_path, index=False)
    print(
        f"  Saved gene ranking to: {rank_path} ({len(gene_rank_df_unique)} unique genes)"
    )
    
    # 保存阴性对照分析结果
    if len(negative_control_results) > 0:
        nc_df = pd.DataFrame(negative_control_results)
        nc_df['ko_gene'] = gene_to_ko
        nc_path = os.path.join(
            OUTPUT_DIR, f"{output_prefix}_{gene_to_ko}_negative_controls.csv"
        )
        nc_df.to_csv(nc_path, index=False)
        print(f"  Saved negative control analysis to: {nc_path}")

    # 保存统计结果（简化版，不再包含置换检验的null分布）
    stats_df = pd.DataFrame(
        {
            "KO_gene": [gene_to_ko],
            "KL_divergence_overall": [dis_overall],
            "KL_divergence_per_gene_mean": [gene_kl_divergences.mean()],
            "KL_divergence_per_gene_std": [gene_kl_divergences.std()],
            "n_target_genes": [len(target_genes)],
            "n_significant_targets": [sum([r['significant'] for r in negative_control_results]) if len(negative_control_results) > 0 else 0],
            "knockout_tissue": ["Brain"],
            "observe_tissue": ["Blood"],
            "n_csf_cells": [n_csf_cells],
            "n_pbmc_cells": [n_pbmc_cells],
        }
    )

    stats_path = os.path.join(
        OUTPUT_DIR, f"{output_prefix}_{gene_to_ko}_statistics.csv"
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved statistics to: {stats_path}")

    # 准备返回结果
    result = {
        "gene_ranking": gene_rank_df,
        "statistics": stats_df,
        "kl_divergence": dis_overall,
        "negative_control_results": negative_control_results,
    }

    # 清理所有中间变量
    del gene_kl_divergences
    gc.collect()

    print(f"\n{'=' * 80}")
    print(f"Completed: {gene_to_ko}")
    print(f"{'=' * 80}\n")

    return result


# ============================================================================
# 主函数
# ============================================================================


def main():
    """依次执行所有组学类型的正向和反向虚拟敲除"""
    global OUTPUT_DIR, DIRECTION, OMICS_LIST

    # 蛋白组专用
    all_omics = ['proteomics']
    
    # 所有方向
    all_directions = ['forward', 'reverse']
    
    print("=" * 80)
    print("GenKI Virtual Knockout Pipeline - 完整流程")
    print("=" * 80)
    print(f"组学类型: {', '.join(all_omics)}")
    print(f"敲除方向: {', '.join(all_directions)}")
    print("=" * 80 + "\n")
    
    for direction in all_directions:
        for omics_name in all_omics:
            DIRECTION = direction
            OMICS_LIST = [omics_name]
            
            # 设置输出目录（用 PROJECT_ROOT 避免 /mnt/d/ 问题）
            if DIRECTION == 'reverse':
                OUTPUT_DIR = str(PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3_reverse")
            else:
                OUTPUT_DIR = str(PROJECT_ROOT / "output" / "step4_virtual_knockout" / "GenKI_NO3")
            
            print("\n" + "=" * 80)
            print(f"开始: {omics_name.upper()} - {DIRECTION.upper()}")
            print("=" * 80)
            print(f"输出目录: {OUTPUT_DIR}")
            print()
            
            try:
                # 创建输出目录
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                print(f"\n{'=' * 80}")
                print(f"Processing omics: {omics_name}")
                print(f"{'=' * 80}")

                # 加载跨组织 Hub 基因（带类别标记）
                hub_genes_with_category, observation_genes = load_cross_tissue_hubs_and_backtrace(omics_name, direction=DIRECTION)

                if not hub_genes_with_category:
                    print(f"  No hub genes found for {omics_name}, skipping...")
                    continue

                print(f"  Found {len(hub_genes_with_category)} hub genes")

                # 提取基因列表（用于数据加载）
                # 注意：这里保留重复是为了兼容之前的数据加载逻辑
                # 重复来自于同一基因同时出现在 core_overlap 和 extended_overlap
                hub_genes = [gene for gene, _ in hub_genes_with_category]
                
                # 合并hub_genes和observation_genes，确保所有边文件中的基因都被保护
                all_protected_genes = list(set(hub_genes + observation_genes))
                print(f"  Total protected genes (hub + observation): {len(all_protected_genes)}")

                # 计算统一的基因列表（在合并数据上计算HVG + 保护targets）
                print(f"\n[Computing common HVG across CSF and PBMC]")
                common_genes = compute_common_hvg(
                    SC_DATA_PATH,
                    tissue_types=["Brain", "Blood"],
                    target_genes=all_protected_genes,
                    n_top_genes=500
                )
                
                if not common_genes:
                    print(f"  ERROR: Failed to compute common genes")
                    continue
                
                print(f"  Final unified gene set: {len(common_genes)} genes")

                # 加载 CSF 和 PBMC 数据（使用统一的基因列表）
                print(f"\n[Loading CSF data]")
                csf_data = load_single_cell_data_h5ad(
                    SC_DATA_PATH,
                    tissue_type="Brain",
                    target_genes=None,  # 不再需要，已在common_genes中
                    max_cells=3000,
                    common_genes=common_genes,  # 传入统一的基因列表
                )

                print(f"\n[Loading PBMC data]")
                pbmc_data = load_single_cell_data_h5ad(
                    SC_DATA_PATH,
                    tissue_type="Blood",
                    target_genes=None,  # 不再需要，已在common_genes中
                    max_cells=3000,
                    common_genes=common_genes,  # 传入统一的基因列表
                )

                if csf_data is None or pbmc_data is None:
                    print(f"  Failed to load data for {omics_name}, skipping...")
                    continue

                # Metacell 聚合（降低维度，让单基因敲除影响更显著）- 带缓存
                print(f"\n[Aggregating to metacells]")
                metacell_cache_path = os.path.join(OUTPUT_DIR, f"{omics_name}_metacells.pkl")

                if os.path.exists(metacell_cache_path):
                    print(f"  Loading cached metacells from: {metacell_cache_path}")
                    with open(metacell_cache_path, 'rb') as f:
                        metacell_cache = pickle.load(f)
                    csf_data = metacell_cache['csf']
                    pbmc_data = metacell_cache['pbmc']
                    print(f"  Loaded CSF: {csf_data.shape}, PBMC: {pbmc_data.shape}")
                else:
                    print(f"  Computing metacells (will cache for future runs)...")
                    csf_data = aggregate_to_metacells(csf_data, n_metacells=200, min_cells_per_metacell=3)
                    pbmc_data = aggregate_to_metacells(pbmc_data, n_metacells=200, min_cells_per_metacell=3)
                    
                    # 保存缓存
                    print(f"  Saving metacells to cache: {metacell_cache_path}")
                    with open(metacell_cache_path, 'wb') as f:
                        pickle.dump({'csf': csf_data, 'pbmc': pbmc_data}, f)
                    print(f"  Cache saved successfully")

                # 处理所有 hub 基因（临时：只处理第1个基因快速验证）
                print(f"\n  Processing all hub genes...")
                
                # ========================================================================
                # 优化：预先构建和缓存WT网络（所有基因共享）
                # ========================================================================
                print(f"\n[Building WT network (shared across all genes)]")
                wt_cache_path = os.path.join(OUTPUT_DIR, f"{omics_name}_wt_cache.pkl")
                
                if os.path.exists(wt_cache_path):
                    print(f"  Loading cached WT network from: {wt_cache_path}")
                    with open(wt_cache_path, 'rb') as f:
                        wt_cache = pickle.load(f)
                    z_mu_wt_shared = wt_cache['z_mu']
                    z_std_wt_shared = wt_cache['z_std']
                    all_genes_list = wt_cache['all_genes']
                    n_csf_cells = wt_cache['n_csf_cells']
                    n_pbmc_cells = wt_cache['n_pbmc_cells']
                    print(f"  Loaded WT latent variables: mu shape={z_mu_wt_shared.shape}, std shape={z_std_wt_shared.shape}")
                else:
                    print(f"  Computing WT network (will cache for future runs)...")
                    
                    # 确保基因顺序一致
                    # 注意：CSF和PBMC已经使用相同的基因集（common_genes），不需要再取交集
                    if not csf_data.index.equals(pbmc_data.index):
                        print(f"  WARNING: Gene order mismatch, reordering...")
                        common_genes_actual = csf_data.index.intersection(pbmc_data.index)
                        csf_data = csf_data.loc[common_genes_actual]
                        pbmc_data = pbmc_data.loc[common_genes_actual]
                    
                    print(f"  Final genes: {len(csf_data.index)}")
                    
                    # 合并CSF和PBMC数据
                    wt_data_df = pd.concat([csf_data, pbmc_data], axis=1)
                    n_csf_cells = csf_data.shape[1]
                    n_pbmc_cells = pbmc_data.shape[1]
                    all_genes_list = wt_data_df.index.tolist()
                    
                    print(f"  WT data shape: {wt_data_df.shape}")
                    print(f"  Cell mask: {n_csf_cells} CSF + {n_pbmc_cells} PBMC")
                    
                    # 构建WT PCNet
                    wt_adj = build_pcnet(
                        wt_data_df, n_comp=N_COMP, scale_scores=SCALE_SCORES, symmetric=SYMMETRIC, q=Q
                    )
                    wt_data = create_pyg_data(wt_data_df, wt_adj)
                    
                    # 清理wt_adj
                    del wt_adj, wt_data_df
                    gc.collect()
                    
                    # 训练WT VGAE
                    print(f"  Training WT VGAE...", flush=True)
                    wt_trainer = VGAE_trainer(
                        data=wt_data,
                        out_channels=OUT_CHANNELS,
                        epochs=EPOCHS,
                        lr=LR,
                        weight_decay=WEIGHT_DECAY,
                        beta=BETA,
                        log_dir=None,
                        verbose=True,  # 优化：主实验显示详细进度
                        seed=RANDOM_STATE,
                    )
                    wt_trainer.train()
                    
                    print(f"  WT VGAE training completed")
                    epoch, loss, auc, ap = wt_trainer.final_metrics
                    print(f"    Final metrics: Loss={loss:.4f}, AUROC={auc:.4f}, AP={ap:.4f}")
                    
                    # 获取WT潜在变量
                    z_mu_wt_shared, z_std_wt_shared = wt_trainer.get_latent_vars(wt_data)
                    print(f"  WT latent variables: mu shape={z_mu_wt_shared.shape}, std shape={z_std_wt_shared.shape}")
                    
                    # 清理trainer和data
                    del wt_trainer, wt_data
                    gc.collect()
                    
                    # 保存缓存
                    print(f"  Saving WT network to cache: {wt_cache_path}")
                    with open(wt_cache_path, 'wb') as f:
                        pickle.dump({
                            'z_mu': z_mu_wt_shared,
                            'z_std': z_std_wt_shared,
                            'all_genes': all_genes_list,
                            'n_csf_cells': n_csf_cells,
                            'n_pbmc_cells': n_pbmc_cells
                        }, f)
                    print(f"  Cache saved successfully")

                all_results = {}

                from tqdm import tqdm
                # 处理所有hub基因
                for gene in tqdm(hub_genes, desc=f"  {omics_name} genes", ncols=80):
                    try:
                        result = virtual_knockout_genki_cross_tissue(
                            csf_data_df=csf_data,
                            pbmc_data_df=pbmc_data,
                            gene_to_ko=gene,
                            output_prefix=f"{omics_name}",
                            z_mu_wt_shared=z_mu_wt_shared,
                            z_std_wt_shared=z_std_wt_shared,
                            all_genes_list=all_genes_list,
                            n_csf_cells=n_csf_cells,
                            n_pbmc_cells=n_pbmc_cells,
                        )

                        if result is not None:
                            all_results[gene] = result

                        # 每个基因处理完后清理内存
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"\nERROR processing {gene}: {str(e)}")
                        import traceback

                        traceback.print_exc()

                        # 错误后也清理内存
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        continue

                # 汇总结果
                print(f"\n{'=' * 80}")
                print(f"Summary for {omics_name}")
                print(f"{'=' * 80}")
                print(f"Successfully processed: {len(all_results)} / {len(hub_genes)} genes")

                for gene, result in all_results.items():
                    print(f"\n{gene}:")
                    print(f"  KL divergence (PBMC): {result['kl_divergence']:.6f}")
                    if 'negative_control_results' in result and len(result['negative_control_results']) > 0:
                        n_sig = sum([r['significant'] for r in result['negative_control_results']])
                        n_total = len(result['negative_control_results'])
                        print(f"  Target genes analyzed: {n_total}, Significant: {n_sig}")
                    print(
                        f"  Top affected gene: {result['gene_ranking'].iloc[0]['Gene']} (KL divergence={result['gene_ranking'].iloc[0]['KL_Divergence']:.4f})"
                    )

                # 清理组学数据
                del csf_data, pbmc_data, all_results
                gc.collect()
                
            except Exception as e:
                print(f"\n❌ {omics_name.upper()} - {DIRECTION.upper()} 执行失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"继续执行下一个任务...\n")
                continue
    
    print("\n" + "=" * 80)
    print("Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
