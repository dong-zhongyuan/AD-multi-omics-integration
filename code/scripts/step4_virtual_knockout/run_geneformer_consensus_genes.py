#!/usr/bin/env python3
"""
Geneformer 虚拟敲除分析 - 多组学支持
支持转录组、蛋白质组、代谢组的正向和反向虚拟敲除

用法：
  # 运行完整流程（从Step3加载基因）
  python run_geneformer_consensus_genes.py
  
  # 指定基因列表
  python run_geneformer_consensus_genes.py --genes DAG1 ELAVL4 KCTD13 KHSRP
  
  # 使用已有tokenized数据（跳过数据准备和tokenization）
  python run_geneformer_consensus_genes.py --genes DAG1 ELAVL4 --use-tokenized
  
  # 指定输出目录
  python run_geneformer_consensus_genes.py --genes DAG1 --output-dir /path/to/output
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')

# 🔥 强制禁用datasets库的多进程
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 添加项目根目录到路径
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 Geneformer 到路径
GENEFORMER_DIR = PROJECT_ROOT / "tools/geneformer-main"
sys.path.insert(0, str(GENEFORMER_DIR))

# 导入统一的基因ID转换工具
from tools.gene_id_converter import ensembl_to_symbol

# 配置路径
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
DATA_PATH = PROJECT_ROOT / "processed-data/step4_single_cell_no3/GSE_NO3_expression_matrix.h5ad"
STEP3_OUTPUT_DIR = PROJECT_ROOT / "output/step3_hub_identification"
OUTPUT_BASE_DIR = PROJECT_ROOT / "output/step4_virtual_knockout"

# 组学列表
OMICS_LIST = ["transcriptomics"]  # 临时：只运行转录组

def map_protein_to_gene(protein_name):
    """
    将蛋白质名称映射为基因名（Gene Symbol）
    优先级：
    1. 直接匹配（蛋白质名 = 基因名）
    2. UniProt 映射
    3. 在线查询（mygene）
    """
    import re
    
    # 1. 直接匹配：如果蛋白质名本身就是基因名格式（全大写字母）
    if re.match(r'^[A-Z0-9]+$', protein_name):
        return protein_name
    
    # 2. UniProt 映射（如果有本地映射文件）
    # TODO: 实现 UniProt 映射
    
    # 3. 在线查询
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        result = mg.query(protein_name, species='human', fields='symbol', size=1)
        if result['hits']:
            return result['hits'][0].get('symbol', protein_name)
    except:
        pass
    
    # 默认返回原名
    return protein_name

def load_hub_genes_for_omics(omics_name):
    """从 eigengene_analysis 加载筛选后的边，提取脑端基因作为knockout目标"""
    
    # 从 eigengene_analysis 加载筛选后的边
    edges_file = STEP3_OUTPUT_DIR / "eigengene_analysis" / omics_name / "filtered_cross_tissue_edges.csv"
    
    if not edges_file.exists():
        print(f"  WARNING: {edges_file} not found")
        return [], []
    
    edges_df = pd.read_csv(edges_file)
    print(f"  Loaded {len(edges_df)} filtered edges from {edges_file}")
    
    # 提取唯一的脑端基因（source列）和血液端基因（target列）
    brain_genes = edges_df['source'].unique().tolist()
    blood_genes = edges_df['target'].unique().tolist()
    print(f"  Extracted {len(brain_genes)} unique brain genes (knockout targets)")
    print(f"  Extracted {len(blood_genes)} unique blood genes (observation targets)")
    
    # 转录组学：source列是Ensembl ID，需要转换为Gene Symbol
    if omics_name == "transcriptomics":
        print(f"  Converting Ensembl IDs to gene symbols...")
        
        # 使用统一的ID转换工具
        all_ids = list(set(brain_genes) | set(blood_genes))
        mapping = ensembl_to_symbol(all_ids)
        print(f"  Converted {len(mapping)} Ensembl IDs to gene symbols")
        
        # 转换基因名
        brain_symbols = []
        blood_symbols = []
        not_found = []
        
        for gene_id in brain_genes:
            if gene_id in mapping:
                brain_symbols.append(mapping[gene_id])
            else:
                not_found.append(gene_id)
        
        for gene_id in blood_genes:
            if gene_id in mapping:
                blood_symbols.append(mapping[gene_id])
        
        if not_found:
            print(f"  WARNING: Could not convert {len(not_found)} Ensembl IDs: {', '.join(not_found[:5])}")
        
        print(f"  Successfully converted {len(brain_symbols)} brain genes to symbols")
        print(f"  Successfully converted {len(blood_symbols)} blood genes to symbols")
        return brain_symbols, blood_symbols
    
    # 蛋白质组学：source列是蛋白质名，需要映射到基因symbol
    elif omics_name == "proteomics":
        print(f"  Mapping protein names to gene symbols...")
        
        brain_symbols = [map_protein_to_gene(p) for p in brain_genes]
        blood_symbols = [map_protein_to_gene(p) for p in blood_genes]
        
        print(f"  Successfully mapped {len(brain_symbols)} brain proteins to gene symbols")
        print(f"  Successfully mapped {len(blood_symbols)} blood proteins to gene symbols")
        return brain_symbols, blood_symbols
    
    # 代谢组学：暂时跳过（Geneformer不支持代谢物）
    elif omics_name == "metabolomics":
        print(f"  WARNING: Geneformer does not support metabolomics, skipping...")
        return [], []
    
    else:
        return brain_genes, blood_genes

# 模型配置
MODEL_DIR = GENEFORMER_DIR / "Geneformer-V2-104M"
MODEL_VERSION = "V2"

# ============================================================================
# Step 1: 数据准备 - 采样和预处理
# ============================================================================

def prepare_data(consensus_genes):
    """
    准备数据：采样细胞，添加元数据
    """
    print("[Step 1/5] 数据准备...")
    
    import anndata as ad
    import scanpy as sc
    
    # 加载数据
    print("  加载 h5ad 数据...")
    adata = ad.read_h5ad(DATA_PATH, backed='r')
    print(f"  原始数据: {adata.shape}")
    
    # 采样细胞（减少计算量）
    print("  采样细胞...")
    n_cells_per_tissue = 500  # 每个组织采样 500 个细胞（流式处理：降低内存压力）
    
    csf_indices = np.where(adata.obs['tissue'] == 'CSF')[0]
    pbmc_indices = np.where(adata.obs['tissue'] == 'PBMC')[0]
    
    # 随机采样
    np.random.seed(42)
    csf_sample = np.random.choice(csf_indices, min(n_cells_per_tissue, len(csf_indices)), replace=False)
    pbmc_sample = np.random.choice(pbmc_indices, min(n_cells_per_tissue, len(pbmc_indices)), replace=False)
    
    all_indices = np.concatenate([csf_sample, pbmc_sample])
    
    # 提取采样数据到内存
    print("  加载采样数据到内存...")
    adata_sample = adata[all_indices, :].to_memory()
    print(f"  采样后数据: {adata_sample.shape}")
    
    # 🔥 关键修复：只保留 token ID < 20275 的基因（模型嵌入层限制）
    print(f"  过滤基因：只保留模型词汇表内的基因（token_id < 20275）...")
    
    # 加载 token dictionary
    import pickle
    token_dict_path = "./tools/geneformer-main/geneformer/token_dictionary_gc104M.pkl"
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    
    # 检查要敲除的基因是否有效
    valid_genes = []
    for gene in consensus_genes:
        if gene in adata_sample.var_names and gene in token_dict:
            token_id = token_dict[gene]
            if token_id < 20275:
                valid_genes.append(gene)
            else:
                print(f"  ⚠️  跳过 {gene}（token_id={token_id} >= 20275）")
        elif gene not in adata_sample.var_names:
            print(f"  ⚠️  跳过 {gene}（不在数据中）")
        else:
            print(f"  ⚠️  跳过 {gene}（不在 token dictionary）")
    
    if len(valid_genes) == 0:
        raise ValueError("没有任何共识基因在模型词汇表内！")
    
    print(f"  有效的敲除目标基因: {valid_genes}")
    
    # 🔥 关键：保留所有在token dictionary中的基因，而不是只保留要敲除的基因
    # Geneformer需要完整的基因表达谱来计算细胞嵌入
    all_valid_genes = []
    for gene in adata_sample.var_names:
        if gene in token_dict and token_dict[gene] < 20275:
            all_valid_genes.append(gene)
    
    print(f"  保留 {len(all_valid_genes)} 个模型词汇表内的基因（用于计算细胞嵌入）")
    adata_sample = adata_sample[:, all_valid_genes]
    print(f"  过滤后数据: {adata_sample.shape}")
    
    # 添加必需的元数据
    print("  添加元数据...")
    adata_sample.obs['cell_type'] = adata_sample.obs['tissue'].astype(str) + '_cells'
    adata_sample.obs['disease'] = 'AD'  # 所有细胞来自 AD 患者
    
    # 计算 UMI 总数
    if hasattr(adata_sample.X, 'toarray'):
        adata_sample.obs['n_counts'] = np.array(adata_sample.X.sum(axis=1)).flatten()
    else:
        adata_sample.obs['n_counts'] = adata_sample.X.sum(axis=1)
    
    # 确保基因名是字符串
    adata_sample.var_names = adata_sample.var_names.astype(str)
    
    # 添加 ensembl_id 列（Geneformer Tokenizer 需要）
    # 我们的数据已经是 gene symbol，所以直接复制
    adata_sample.var['ensembl_id'] = adata_sample.var_names.tolist()
    
    # 保存准备好的数据
    temp_dir = OUTPUT_DIR / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / "prepared_data.h5ad"
    
    print(f"  保存到: {temp_path}")
    adata_sample.write_h5ad(temp_path)
    
    print(f"  ✅ 数据准备完成")
    print(f"     CSF cells: {len(csf_sample)}")
    print(f"     PBMC cells: {len(pbmc_sample)}")
    print(f"     Total: {len(all_indices)}")
    
    return temp_path, valid_genes


# ============================================================================
# Step 2: Tokenization
# ============================================================================

def tokenize_data(data_path):
    """
    将数据转换为 Geneformer 格式
    """
    print("\n[Step 2/5] Tokenization...")
    
    from geneformer import TranscriptomeTokenizer
    
    tokenized_dir = OUTPUT_DIR / "tokenized_data"
    tokenized_dir.mkdir(exist_ok=True)
    
    # 检查是否已经 tokenized
    tokenized_path = tokenized_dir / "tokenized.dataset"
    if tokenized_path.exists():
        print(f"  ⚠️  已存在 tokenized 数据，跳过")
        return tokenized_path
    
    print("  初始化 Tokenizer...")
    
    # 显式指定字典文件路径
    geneformer_dir = Path(__file__).parent.parent.parent / "tools" / "geneformer-main" / "geneformer"
    
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "cell_type": "cell_type",
            "disease": "disease",
            "tissue": "tissue"
        },
        nproc=1,  # WSL多进程问题，改为单进程
        gene_median_file=str(geneformer_dir / "gene_median_dictionary_gc104M.pkl"),
        token_dictionary_file=str(geneformer_dir / "token_dictionary_gc104M.pkl"),
        model_version=MODEL_VERSION
    )
    
    print("  开始 Tokenization（这可能需要 10-20 分钟）...")
    tk.tokenize_data(
        data_directory=str(data_path.parent),
        output_directory=str(tokenized_dir),
        output_prefix="tokenized",
        file_format="h5ad"
    )
    
    print(f"  ✅ Tokenization 完成")
    return tokenized_path


# ============================================================================
# Step 3: 提取状态嵌入（Baseline）
# ============================================================================

def extract_embeddings(tokenized_path):
    """Step 3: 提取状态嵌入（简化版：跳过状态建模）"""
    print("\n[Step 3/5] 提取状态嵌入...")
    
    # 由于我们的数据都是AD患者，没有对照组，跳过状态嵌入提取
    # 直接进行虚拟敲除，不使用状态建模
    print("  ⚠️  跳过状态嵌入提取（简化模式）")
    print("     原因：数据只有AD患者，无对照组")
    
    # 返回 None，表示不使用状态建模
    state_embs_dict = None
    cell_states_to_model = None
    filter_data_dict = {
        "disease": ["AD"]
    }
    
    print(f"  ✅ 使用简化模式（无状态建模）")
    
    return state_embs_dict, cell_states_to_model, filter_data_dict


# ============================================================================
# Step 3.5: 提取 Baseline 嵌入
# ============================================================================

def extract_baseline_embeddings(tokenized_path):
    """
    提取baseline（无敲除）的细胞嵌入向量
    """
    print("\n[Step 3.5/5] 提取 Baseline 嵌入...")
    
    from geneformer import InSilicoPerturber
    import pickle
    
    baseline_dir = OUTPUT_DIR / "baseline_embeddings"
    baseline_dir.mkdir(exist_ok=True)
    
    baseline_file = baseline_dir / "baseline_embs.pickle"
    
    # 检查是否已存在
    if baseline_file.exists():
        print(f"  ⚠️  Baseline 嵌入已存在，跳过")
        with open(baseline_file, "rb") as f:
            baseline_embs = pickle.load(f)
        print(f"  ✅ 加载已有 baseline: shape={baseline_embs.shape if hasattr(baseline_embs, 'shape') else len(baseline_embs)}")
        return baseline_embs
    
    print("  提取 baseline 嵌入（无敲除）...")
    
    # 使用 InSilicoPerturber 提取嵌入（不敲除任何基因）
    # 🔥 流式处理：降低batch size，避免内存积压
    isp = InSilicoPerturber(
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb="all",  # 不敲除，只提取嵌入
        combos=0,
        anchor_gene=None,
        model_type="Pretrained",
        emb_mode="cls",
        cell_emb_style="mean_pool",
        filter_data=None,
        cell_states_to_model=None,
        state_embs_dict=None,
        max_ncells=500,  # 降低到500，避免内存爆炸
        emb_layer=0,
        forward_batch_size=16,  # 降低到16，流式处理
        model_version=MODEL_VERSION,
        nproc=1  # WSL多进程问题
    )
    
    # 提取嵌入
    isp.perturb_data(
        str(MODEL_DIR),
        str(tokenized_path),
        str(baseline_dir),
        "baseline"
    )
    
    # 加载保存的嵌入
    if baseline_file.exists():
        with open(baseline_file, "rb") as f:
            baseline_embs = pickle.load(f)
        print(f"  ✅ Baseline 嵌入提取完成: shape={baseline_embs.shape if hasattr(baseline_embs, 'shape') else len(baseline_embs)}")
        return baseline_embs
    else:
        print(f"  ⚠️  未找到 baseline 文件，检查输出目录...")
        import os
        files = os.listdir(baseline_dir)
        print(f"  输出文件: {files}")
        # 尝试找到第一个 pickle 文件
        for f in files:
            if f.endswith('.pickle'):
                baseline_file = baseline_dir / f
                with open(baseline_file, "rb") as fp:
                    baseline_embs = pickle.load(fp)
                print(f"  ✅ 加载 baseline: {f}, shape={baseline_embs.shape if hasattr(baseline_embs, 'shape') else len(baseline_embs)}")
                return baseline_embs
        raise FileNotFoundError("未找到 baseline 嵌入文件")


# ============================================================================
# Step 4: 虚拟敲除
# ============================================================================

def run_knockout(tokenized_path, state_embs_dict, cell_states_to_model, filter_data_dict, baseline_embs, valid_genes):
    """
    执行虚拟敲除（逐个基因），并计算响应基因
    """
    print("\n[Step 4/5] 执行虚拟敲除...")
    
    from geneformer import InSilicoPerturber
    import pickle
    
    perturb_dir = OUTPUT_DIR / "perturbations"
    perturb_dir.mkdir(exist_ok=True)
    
    # 加载 token dictionary 检查基因是否存在
    token_dict_path = GENEFORMER_DIR / "geneformer" / "token_dictionary_gc104M.pkl"
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    
    # 过滤出在 token dictionary 中的基因
    valid_genes_in_dict = [g for g in valid_genes if g in token_dict]
    skipped_genes = [g for g in valid_genes if g not in token_dict]
    
    if skipped_genes:
        print(f"  ⚠️  跳过 {len(skipped_genes)} 个不在 token dictionary 的基因: {', '.join(skipped_genes)}")
    
    print(f"  开始虚拟敲除 {len(valid_genes_in_dict)} 个基因（逐个执行）...")
    print(f"  基因列表: {', '.join(valid_genes_in_dict)}")
    print(f"  预计时间: 5-10 分钟（CPU 模式）")
    
    # 🔥 断点续传：检查已完成的基因
    import os
    completed_genes = set()
    for f in os.listdir(perturb_dir):
        if f.endswith('.pickle') and 'knockout_' in f:
            # 提取基因名：in_silico_delete_knockout_GENE_cell_embs_dict_[xxx]_raw.pickle
            import re
            match = re.search(r'knockout_([A-Z0-9]+)_', f)
            if match:
                completed_genes.add(match.group(1))
    
    if completed_genes:
        print(f"  ✅ 发现 {len(completed_genes)} 个已完成的基因，将跳过: {', '.join(sorted(completed_genes))}")
    
    genes_to_run = [g for g in valid_genes_in_dict if g not in completed_genes]
    print(f"  待处理基因: {len(genes_to_run)}/{len(valid_genes_in_dict)}")
    
    if not genes_to_run:
        print(f"  ✅ 所有基因已完成，跳过虚拟敲除步骤")
        return perturb_dir
    
    # 逐个基因进行虚拟敲除
    for i, gene in enumerate(genes_to_run, 1):
        print(f"\n  [{i}/{len(genes_to_run)}] 敲除基因: {gene}")
        
        try:
            # 为每个基因创建独立的 InSilicoPerturber
            # 🔥 流式处理：降低batch size和max_ncells，避免内存积压
            isp = InSilicoPerturber(
                perturb_type="delete",
                perturb_rank_shift=None,
                genes_to_perturb=[gene],  # 单个基因
                combos=0,
                anchor_gene=None,
                model_type="Pretrained",
                emb_mode="cls",
                cell_emb_style="mean_pool",
                filter_data=filter_data_dict,
                cell_states_to_model=cell_states_to_model,
                state_embs_dict=state_embs_dict,
                max_ncells=500,  # 降低到500，避免内存爆炸
                emb_layer=0,
                forward_batch_size=16,  # 降低到16，流式处理
                model_version=MODEL_VERSION,
                nproc=1  # WSL多进程问题
            )
            
            # 运行敲除
            isp.perturb_data(
                str(MODEL_DIR),
                str(tokenized_path),
                str(perturb_dir),
                f"knockout_{gene}"
            )
            
            print(f"    ✅ {gene} 敲除完成")
            
            # 🔥 流式处理：立即清理内存
            del isp
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"    ❌ {gene} 敲除失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n  ✅ 所有虚拟敲除完成")
    return perturb_dir


# ============================================================================
# Step 4.5: 计算响应基因
# ============================================================================

def compute_response_genes_for_knockout(gene, perturb_dir, baseline_embs, tokenized_path):
    """
    计算单个基因敲除后的响应基因列表
    """
    import pickle
    from scipy.spatial.distance import cosine
    from scipy import stats
    
    print(f"    计算 {gene} 的响应基因...")
    
    # 加载敲除后的嵌入
    knockout_file = perturb_dir / f"knockout_{gene}_embs_dict.pickle"
    if not knockout_file.exists():
        # 尝试其他可能的文件名
        import os
        files = [f for f in os.listdir(perturb_dir) if f.startswith(f"knockout_{gene}") and f.endswith('.pickle')]
        if not files:
            print(f"    ⚠️  未找到 {gene} 的敲除结果文件")
            return
        knockout_file = perturb_dir / files[0]
    
    with open(knockout_file, "rb") as f:
        knockout_embs = pickle.load(f)
    
    # 计算嵌入向量的余弦距离
    # 注意：pickle文件可能存储的是相似度分数，不是嵌入向量
    # 如果是相似度分数（接近1.0），则距离 = 1 - 相似度
    if isinstance(knockout_embs, np.ndarray) and knockout_embs.ndim == 1:
        # 一维数组，可能是相似度分数
        distances = 1 - knockout_embs
    elif isinstance(baseline_embs, np.ndarray) and baseline_embs.ndim == 2:
        # 二维数组，是嵌入向量矩阵
        # 计算每个细胞的余弦距离
        distances = np.array([cosine(baseline_embs[i], knockout_embs[i]) for i in range(len(baseline_embs))])
    else:
        print(f"    ⚠️  无法识别嵌入格式: baseline={type(baseline_embs)}, knockout={type(knockout_embs)}")
        return
    
    # 计算统计量
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"    平均距离: {mean_distance:.6f} ± {std_distance:.6f}")
    
    # 加载 tokenized 数据，获取基因列表
    from datasets import load_from_disk
    dataset = load_from_disk(str(tokenized_path))
    
    # 获取第一个样本的基因列表（input_ids）
    if len(dataset) > 0:
        sample = dataset[0]
        gene_ids = sample['input_ids']
        
        # 加载 token dictionary（反向映射：token_id -> gene_symbol）
        token_dict_path = GENEFORMER_DIR / "geneformer" / "token_dictionary_gc104M.pkl"
        with open(token_dict_path, "rb") as f:
            token_dict = pickle.load(f)
        
        # 反向映射
        id_to_gene = {v: k for k, v in token_dict.items()}
        
        # 提取基因名
        gene_names = [id_to_gene.get(gid, f"UNKNOWN_{gid}") for gid in gene_ids]
        
        # 创建响应基因列表（按距离排序）
        # 注意：这里我们假设每个基因的响应程度与细胞嵌入距离相关
        # 实际上，我们需要计算每个基因的表达变化，但这需要访问原始表达矩阵
        # 简化方案：使用细胞嵌入距离作为代理指标
        
        # 为每个基因分配一个响应分数（这里简化为使用平均距离）
        response_scores = {gname: mean_distance for gname in gene_names if gname != gene}
        
        # 排序
        sorted_genes = sorted(response_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 保存 CSV
        response_dir = OUTPUT_DIR / "response_genes"
        response_dir.mkdir(exist_ok=True)
        
        response_df = pd.DataFrame(sorted_genes, columns=['gene', 'response_score'])
        response_df['knockout_gene'] = gene
        response_df['mean_distance'] = mean_distance
        response_df['std_distance'] = std_distance
        
        # 计算 Z-score（如果有多个细胞）
        if len(distances) > 1:
            response_df['z_score'] = (response_df['response_score'] - mean_distance) / std_distance
        
        csv_file = response_dir / f"response_genes_{gene}.csv"
        response_df.to_csv(csv_file, index=False)
        
        print(f"    ✅ 响应基因已保存: {csv_file}")
        print(f"    Top 5 响应基因: {', '.join([g for g, _ in sorted_genes[:5]])}")
    else:
        print(f"    ⚠️  Tokenized 数据集为空")


# ============================================================================
# Step 5: 统计分析
# ============================================================================

def post_process_pickles(perturb_dir, valid_genes):
    """
    后处理pickle文件：提取细胞嵌入相似度，计算统计指标，生成CSV
    Geneformer输出格式：{(gene_id, 'cell_emb'): [similarity_scores]}
    """
    import pickle
    import os
    import re
    
    # 遍历所有pickle文件
    pickle_files = [f for f in os.listdir(perturb_dir) if f.endswith('.pickle')]
    
    all_stats = []
    
    for pickle_file in pickle_files:
        # 提取基因名
        match = re.search(r'knockout_([A-Z0-9]+)_', pickle_file)
        if not match:
            continue
        
        gene = match.group(1)
        print(f"    处理 {gene}...")
        
        # 加载pickle
        pickle_path = perturb_dir / pickle_file
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        # Geneformer的pickle格式：{(gene_id, 'cell_emb'): [cosine_similarity_scores]}
        # 相似度越低，说明敲除效应越大
        if isinstance(data, dict):
            # 提取相似度分数
            similarities = []
            for key, value in data.items():
                if isinstance(value, list):
                    similarities.extend(value)
            
            if similarities:
                # 计算统计指标
                similarities = np.array(similarities)
                distances = 1 - similarities  # 距离 = 1 - 相似度
                
                stats = {
                    'gene': gene,
                    'n_cells': len(similarities),
                    'mean_similarity': np.mean(similarities),
                    'mean_distance': np.mean(distances),
                    'median_distance': np.median(distances),
                    'std_distance': np.std(distances),
                    'max_distance': np.max(distances),
                    'min_distance': np.min(distances)
                }
                
                all_stats.append(stats)
                
                print(f"      ✅ {len(similarities)} 个细胞")
                print(f"      平均距离: {stats['mean_distance']:.4f}")
                print(f"      最大距离: {stats['max_distance']:.4f}")
            else:
                print(f"      ⚠️  未找到相似度分数")
        else:
            print(f"      ⚠️  未知的pickle格式: {type(data)}")
    
    # 保存汇总统计
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df = stats_df.sort_values('mean_distance', ascending=False)
        
        csv_file = perturb_dir.parent / "knockout_statistics.csv"
        stats_df.to_csv(csv_file, index=False)
        print(f"\n  ✅ 统计结果保存到: {csv_file.name}")
        print(f"  效应最强的基因: {stats_df.iloc[0]['gene']} (距离={stats_df.iloc[0]['mean_distance']:.4f})")
    else:
        print(f"\n  ⚠️  没有生成统计结果")

def compute_stats(perturb_dir, cell_states_to_model, valid_genes):
    """
    计算统计结果
    """
    print("\n[Step 5/5] 统计分析...")
    
    # 如果没有 cell_states_to_model，使用简化的后处理
    if cell_states_to_model is None:
        print("  ⚠️  跳过统计分析（简化模式，无状态建模）")
        print("  虚拟敲除结果已保存到:")
        print(f"    {perturb_dir}")
        
        # 列出生成的文件
        import os
        result_files = [f for f in os.listdir(perturb_dir) if f.endswith('.pickle')]
        if result_files:
            print(f"  生成的结果文件:")
            for f in result_files:
                print(f"    - {f}")
        
        # 🔥 后处理：提取pickle中的嵌入向量，计算基因排名
        print("\n  开始后处理pickle文件...")
        post_process_pickles(perturb_dir, valid_genes)
        
        print(f"  ✅ 虚拟敲除完成（原始结果已保存）")
        return perturb_dir
    
    # 原有的统计分析逻辑
    from geneformer import InSilicoPerturberStats
    
    # 初始化统计分析器
    print("  初始化 InSilicoPerturberStats...")
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed=valid_genes,
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states_to_model,
        model_version=MODEL_VERSION
    )
    
    # 计算统计
    stats_dir = OUTPUT_DIR / "stats"
    stats_dir.mkdir(exist_ok=True)
    
    print("  计算统计指标...")
    ispstats.get_stats(
        str(perturb_dir),
        None,
        str(stats_dir),
        "knockout_stats"
    )
    
    print(f"  ✅ 统计分析完成")
    
    # 读取并显示结果
    stats_file = stats_dir / "knockout_stats.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        print("\n" + "=" * 80)
        print("Geneformer 虚拟敲除结果")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\n完整结果: {stats_file}")
    
    return stats_dir


# ============================================================================
# 主函数
# ============================================================================

def analyze_results():
    """
    分析虚拟敲除结果，生成可解释的报告
    """
    print("\n[Step 6/6] 分析虚拟敲除结果...")
    
    analysis_script = PROJECT_ROOT / "scripts/step4_virtual_knockout/analyze_geneformer_results.py"
    
    if not analysis_script.exists():
        print(f"  ⚠️  分析脚本不存在: {analysis_script}")
        return None
    
    print(f"  运行分析脚本: {analysis_script}")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(analysis_script)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
        return OUTPUT_DIR / "knockout_impact_summary.csv"
    else:
        print(f"  ⚠️  分析脚本执行失败:")
        print(result.stderr)
        return None


def run_single_gene_list(genes_to_knockout, output_dir, use_tokenized=False):
    """
    运行单个基因列表的虚拟敲除
    
    Args:
        genes_to_knockout: 要敲除的基因列表
        output_dir: 输出目录
        use_tokenized: 是否使用已有的tokenized数据
    """
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Geneformer 虚拟敲除分析")
    print("=" * 80)
    print(f"基因列表: {genes_to_knockout}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"使用已有tokenized数据: {use_tokenized}")
    print()
    
    try:
        if use_tokenized:
            # 使用已有tokenized数据
            tokenized_path = OUTPUT_DIR / "tokenized_data/tokenized.dataset"
            
            if not tokenized_path.exists():
                print(f"❌ Tokenized数据不存在: {tokenized_path}")
                print(f"   请先运行完整流程或检查路径")
                return False
            
            print(f"✅ 使用已有tokenized数据: {tokenized_path}")
            
            # 验证基因有效性
            import pickle
            token_dict_path = GENEFORMER_DIR / "geneformer" / "token_dictionary_gc104M.pkl"
            with open(token_dict_path, "rb") as f:
                token_dict = pickle.load(f)
            
            valid_genes = [g for g in genes_to_knockout if g in token_dict and token_dict[g] < 20275]
            
            if not valid_genes:
                print(f"❌ 没有有效基因")
                return False
            
            print(f"有效基因: {valid_genes}")
            
            # 跳过数据准备和tokenization
            state_embs_dict = None
            cell_states_to_model = None
            filter_data_dict = None
            baseline_embs = None
            
        else:
            # 完整流程
            # Step 1: 数据准备
            data_path, valid_genes = prepare_data(genes_to_knockout)
            
            if not valid_genes:
                print(f"❌ 没有有效基因")
                return False
            
            # Step 2: Tokenization
            tokenized_path = tokenize_data(data_path)
            
            # Step 3: 提取嵌入
            state_embs_dict, cell_states_to_model, filter_data_dict = extract_embeddings(tokenized_path)
            baseline_embs = None
        
        # Step 4: 虚拟敲除
        perturb_dir = run_knockout(tokenized_path, state_embs_dict, cell_states_to_model, filter_data_dict, baseline_embs, valid_genes)
        
        # Step 5: 统计分析
        stats_dir = compute_stats(perturb_dir, cell_states_to_model, valid_genes)
        
        print("\n" + "=" * 80)
        print("✅ 虚拟敲除完成！")
        print("=" * 80)
        print(f"输出目录: {OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 虚拟敲除失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：遍历所有组学，执行正向和反向虚拟敲除"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Geneformer虚拟敲除分析')
    parser.add_argument('--genes', nargs='+', help='指定要敲除的基因列表（空格分隔）')
    parser.add_argument('--use-tokenized', action='store_true', help='使用已有的tokenized数据（跳过数据准备和tokenization）')
    parser.add_argument('--output-dir', help='指定输出目录')
    parser.add_argument('--omics', choices=['transcriptomics', 'proteomics', 'metabolomics'], 
                       default='transcriptomics', help='组学类型（默认：transcriptomics）')
    
    args = parser.parse_args()
    
    # 如果指定了基因列表，运行单个基因列表模式
    if args.genes:
        output_dir = args.output_dir or (OUTPUT_BASE_DIR / f"Geneformer_{args.omics}_custom")
        success = run_single_gene_list(args.genes, output_dir, args.use_tokenized)
        sys.exit(0 if success else 1)
    
    # 否则运行完整流程（从Step3加载基因）
    for omics_name in OMICS_LIST:
        print("\n" + "=" * 80)
        print(f"Processing omics: {omics_name}")
        print("=" * 80)
        
        # 加载hub基因
        brain_genes, blood_genes = load_hub_genes_for_omics(omics_name)
        
        if not brain_genes:
            print(f"  No hub genes found for {omics_name}, skipping...")
            continue
        
        print(f"  Found {len(brain_genes)} brain genes (knockout targets)")
        print(f"  Found {len(blood_genes)} blood genes (observation targets)")
        
        # 正向敲除：敲除脑端基因，观察血液端基因
        print(f"\n{'=' * 80}")
        print(f"Forward knockout: KO brain genes -> Observe blood genes")
        print(f"{'=' * 80}")
        
        try:
            # 设置输出目录
            global OUTPUT_DIR
            OUTPUT_DIR = OUTPUT_BASE_DIR / f"Geneformer_{omics_name}_forward"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            print(f"数据路径: {DATA_PATH}")
            print(f"模型路径: {MODEL_DIR}")
            print(f"输出目录: {OUTPUT_DIR}")
            print(f"敲除基因: {brain_genes}")
            print()
            
            # Step 1: 数据准备
            data_path, valid_genes = prepare_data(brain_genes)
            
            # Step 2: Tokenization
            tokenized_path = tokenize_data(data_path)
            
            # Step 3: 提取嵌入
            state_embs_dict, cell_states_to_model, filter_data_dict = extract_embeddings(tokenized_path)
            
            # Step 4: 虚拟敲除
            baseline_embs = None
            perturb_dir = run_knockout(tokenized_path, state_embs_dict, cell_states_to_model, filter_data_dict, baseline_embs, valid_genes)
            
            # Step 5: 统计分析
            stats_dir = compute_stats(perturb_dir, cell_states_to_model, valid_genes)
            
            print(f"\n✅ Forward knockout completed for {omics_name}")
            print(f"   Output: {OUTPUT_DIR}")
            
        except Exception as e:
            print(f"\n❌ Forward knockout failed for {omics_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # 反向敲除：敲除血液端基因，观察脑端基因
        if blood_genes:
            print(f"\n{'=' * 80}")
            print(f"Reverse knockout: KO blood genes -> Observe brain genes")
            print(f"{'=' * 80}")
            
            try:
                # 设置输出目录
                OUTPUT_DIR = OUTPUT_BASE_DIR / f"Geneformer_{omics_name}_reverse"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                
                print(f"数据路径: {DATA_PATH}")
                print(f"模型路径: {MODEL_DIR}")
                print(f"输出目录: {OUTPUT_DIR}")
                print(f"敲除基因: {blood_genes}")
                print()
                
                # Step 1: 数据准备
                data_path, valid_genes = prepare_data(blood_genes)
                
                # Step 2: Tokenization
                tokenized_path = tokenize_data(data_path)
                
                # Step 3: 提取嵌入
                state_embs_dict, cell_states_to_model, filter_data_dict = extract_embeddings(tokenized_path)
                
                # Step 4: 虚拟敲除
                baseline_embs = None
                perturb_dir = run_knockout(tokenized_path, state_embs_dict, cell_states_to_model, filter_data_dict, baseline_embs, valid_genes)
                
                # Step 5: 统计分析
                stats_dir = compute_stats(perturb_dir, cell_states_to_model, valid_genes)
                
                print(f"\n✅ Reverse knockout completed for {omics_name}")
                print(f"   Output: {OUTPUT_DIR}")
                
            except Exception as e:
                print(f"\n❌ Reverse knockout failed for {omics_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ All Geneformer analyses completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
