import os
#!/usr/bin/env python3
"""
Step 0.3: 代谢组数据预处理 (HVG + PCA 降维版)
输入: Metabolomics Workbench 格式（血浆 ST000046 + CSF ST000047）
输出: AnnData 格式 .h5ad 文件（降维到50维）
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from sklearn.decomposition import PCA
import pickle
import sys

# 导入配置管理器
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config
config = get_config()

# 路径配置
DATA_DIR = config.get_path("paths.data_dir")
OUTPUT_DIR = config.get_path("paths.processed_data_dir")
OUTPUT_DIR.mkdir(exist_ok=True)

def read_metabolomics_workbench(data_file):
    """
    读取 Metabolomics Workbench 数据
    正确处理 Samples/Factors 两行格式
    """
    print(f"📖 读取 {data_file.name}...")
    
    # 读取文件
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # 找到数据表起始行
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('MS_METABOLITE_DATA_START'):
            data_start = i + 1
            break
    
    if data_start == 0:
        raise ValueError("未找到 MS_METABOLITE_DATA_START 标记")
    
    # 读取样本名称行（第一行）
    sample_line = lines[data_start].strip().split('\t')
    sample_ids = sample_line[1:]  # 跳过第一列 "Samples"
    
    # 读取 Factors 行（第二行）提取诊断信息
    factors_line = lines[data_start + 1].strip().split('\t')
    factors = factors_line[1:]  # 跳过第一列 "Factors"
    
    # 解析诊断信息（格式: "Cognitive Status:AD"）
    diagnosis = []
    for factor in factors:
        if ':' in factor:
            diagnosis.append(factor.split(':')[1])  # 提取 AD/CN/MCI
        else:
            diagnosis.append('Unknown')
    
    # 从第三行开始读取代谢物数据
    metabolite_data = []
    metabolite_names = []
    
    n_samples = len(sample_ids)
    
    for line in lines[data_start + 2:]:  # 跳过 Samples 和 Factors 行
        if line.startswith('MS_METABOLITE_DATA_END'):
            break
        
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        
        metabolite_name = parts[0]
        values = parts[1:]
        
        # 转换为数值，空值设为 NaN
        # 确保每行都有 n_samples 个值
        numeric_values = []
        for i in range(n_samples):
            if i < len(values):
                v = values[i]
                try:
                    numeric_values.append(float(v) if v.strip() else np.nan)
                except ValueError:
                    numeric_values.append(np.nan)
            else:
                numeric_values.append(np.nan)
        
        metabolite_names.append(metabolite_name)
        metabolite_data.append(numeric_values)
    
    # 构建 DataFrame（样本 × 代谢物）
    df = pd.DataFrame(
        data=np.array(metabolite_data).T,  # 转置：样本在行，代谢物在列
        index=sample_ids,
        columns=metabolite_names
    )
    
    # 构建诊断信息 DataFrame
    diagnosis_df = pd.DataFrame({
        'diagnosis': diagnosis
    }, index=sample_ids)
    
    print(f"  ✅ 读取完成: {df.shape[0]} 样本 × {df.shape[1]} 代谢物")
    print(f"  诊断分布: {diagnosis_df['diagnosis'].value_counts().to_dict()}")
    return df, diagnosis_df

def preprocess_metabolomics(df, min_valid_ratio=0.5):
    """
    代谢组数据预处理
    1. 过滤缺失值过多的代谢物
    2. 填充缺失值（最小值的一半）
    3. Log 转换
    4. Z-score 标准化
    """
    print(f"🔧 数据预处理...")
    print(f"  原始: {df.shape[0]} 样本 × {df.shape[1]} 代谢物")
    
    # 过滤缺失值过多的代谢物
    valid_ratio = df.notna().sum() / len(df)
    valid_metabolites = valid_ratio[valid_ratio >= min_valid_ratio].index
    df_filtered = df[valid_metabolites]
    print(f"  过滤后: {df_filtered.shape[0]} 样本 × {df_filtered.shape[1]} 代谢物")
    print(f"    (移除 {df.shape[1] - df_filtered.shape[1]} 个缺失值过多的代谢物)")
    
    # 填充缺失值（用每个代谢物的最小值的一半）
    for col in df_filtered.columns:
        min_val = df_filtered[col].min()
        if pd.isna(min_val) or min_val <= 0:
            min_val = 1e-6
        df_filtered[col].fillna(min_val / 2, inplace=True)
    
    # Log 转换（加 1 避免 log(0)）
    df_log = np.log2(df_filtered + 1)
    
    # Z-score 标准化（按代谢物）
    df_zscore = (df_log - df_log.mean()) / df_log.std()
    
    print(f"  ✅ 标准化完成")
    return df_zscore

def apply_common_metabolites_pca(adata_plasma, adata_csf, pca_dim=100, output_dir=None):
    """
    先筛选共同代谢物，再 PCA 降维
    
    参数:
        adata_plasma: 血浆 AnnData
        adata_csf: CSF AnnData
        pca_dim: PCA 降维目标维度（默认 100）
        output_dir: 输出目录
    
    返回:
        adata_plasma_pca, adata_csf_pca, common_metabolites
    """
    print(f"🔍 筛选共同代谢物...")
    
    # 找共同代谢物
    common_metabolites = list(set(adata_plasma.var.index) & set(adata_csf.var.index))
    print(f"  共同代谢物数: {len(common_metabolites)}")
    
    # 筛选
    adata_plasma_common = adata_plasma[:, common_metabolites].copy()
    adata_csf_common = adata_csf[:, common_metabolites].copy()
    
    print(f"  血浆: {adata_plasma_common.shape}")
    print(f"  CSF:  {adata_csf_common.shape}")
    
    # PCA 降维
    print(f"🔧 PCA 降维到 {pca_dim} 维...")
    
    # 合并数据进行 PCA
    X_combined = np.vstack([adata_plasma_common.X, adata_csf_common.X])
    
    pca = PCA(n_components=pca_dim, random_state=42)
    X_combined_pca = pca.fit_transform(X_combined)
    
    # 分割回血浆和 CSF
    n_plasma = adata_plasma_common.n_obs
    X_plasma_pca = X_combined_pca[:n_plasma]
    X_csf_pca = X_combined_pca[n_plasma:]
    
    # 创建新的 AnnData
    adata_plasma_pca = ad.AnnData(
        X=X_plasma_pca,
        obs=adata_plasma_common.obs.copy(),
        var=pd.DataFrame(index=[f'PC{i+1}' for i in range(pca_dim)])
    )
    
    adata_csf_pca = ad.AnnData(
        X=X_csf_pca,
        obs=adata_csf_common.obs.copy(),
        var=pd.DataFrame(index=[f'PC{i+1}' for i in range(pca_dim)])
    )
    
    # 保存 PCA loadings 到 uns（用于回溯到原始特征）
    # pca.components_ shape: (n_components, n_features) = (50, 494)
    # 每一行是一个 PC，每一列是一个原始特征的贡献
    adata_plasma_pca.uns['pca_loadings'] = pca.components_
    adata_plasma_pca.uns['pca_feature_names'] = common_metabolites
    adata_csf_pca.uns['pca_loadings'] = pca.components_
    adata_csf_pca.uns['pca_feature_names'] = common_metabolites
    
    # 保存 PCA 模型和共同代谢物列表
    if output_dir:
        pca_file = output_dir / "pca_model_common.pkl"
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  💾 保存 PCA 模型: {pca_file}")
        
        # 保存共同代谢物列表
        common_file = output_dir / "common_metabolites.txt"
        with open(common_file, 'w') as f:
            for metabolite in common_metabolites:
                f.write(f"{metabolite}\n")
        print(f"  💾 保存共同代谢物列表: {common_file}")
        
        # 保存解释方差
        var_explained = pca.explained_variance_ratio_
        print(f"  📊 前 {pca_dim} 个 PC 解释方差: {var_explained.sum():.2%}")
        print(f"     PC1-10: {var_explained[:10].sum():.2%}")
    
    return adata_plasma_pca, adata_csf_pca, common_metabolites
    """
    应用 HVG + PCA 降维（参考 HepaWorld 方法）
    
    步骤：
    1. 计算高变代谢物（HVM, Highly Variable Metabolites）
    2. 在 HVM 空间做 PCA 降维
    3. 保存 PCA 模型和 loading matrix（用于后续反向映射）
    """
    print(f"\n🔬 应用 HVG + PCA 降维...")
    print(f"  目标: {n_top_hvg} 个高变代谢物 → PCA {pca_dim} 维")
    
    # 找到共同代谢物
    common_metabolites = list(set(adata_plasma.var.index) & set(adata_csf.var.index))
    print(f"  共同代谢物: {len(common_metabolites)} 个")
    
    # 提取共同代谢物数据
    plasma_common = adata_plasma[:, common_metabolites].copy()
    csf_common = adata_csf[:, common_metabolites].copy()
    
    # 合并数据用于计算方差（血浆 + CSF）
    combined_X = np.vstack([plasma_common.X, csf_common.X])
    
    # 计算每个代谢物的方差（HVG 选择标准）
    variances = np.var(combined_X, axis=0)
    
    # 选择 top N 高变代谢物
    n_top_hvg = min(n_top_hvg, len(common_metabolites))
    top_indices = np.argsort(variances)[::-1][:n_top_hvg]
    hvg_metabolites = [common_metabolites[i] for i in top_indices]
    
    print(f"  ✅ 选择了 {len(hvg_metabolites)} 个高变代谢物")
    print(f"     方差范围: {variances[top_indices].min():.4f} - {variances[top_indices].max():.4f}")
    
    # 提取 HVG 数据
    plasma_hvg = plasma_common[:, hvg_metabolites].copy()
    csf_hvg = csf_common[:, hvg_metabolites].copy()
    
    # PCA 降维（在血浆数据上训练，应用到两个数据集）
    pca_dim = min(pca_dim, plasma_hvg.n_obs, len(hvg_metabolites))
    print(f"  🔧 PCA 降维到 {pca_dim} 维...")
    
    pca = PCA(n_components=pca_dim, random_state=42)
    plasma_pca = pca.fit_transform(plasma_hvg.X)
    csf_pca = pca.transform(csf_hvg.X)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  ✅ PCA 完成，解释方差: {explained_var:.2%}")
    
    # 创建降维后的 AnnData
    adata_plasma_pca = ad.AnnData(
        X=plasma_pca,
        obs=plasma_hvg.obs.copy(),
        var=pd.DataFrame(index=[f'PC{i+1}' for i in range(pca_dim)])
    )
    
    adata_csf_pca = ad.AnnData(
        X=csf_pca,
        obs=csf_hvg.obs.copy(),
        var=pd.DataFrame(index=[f'PC{i+1}' for i in range(pca_dim)])
    )
    
    # 保存 PCA 模型和元数据
    if output_dir is not None:
        pca_info = {
            'pca_model': pca,
            'hvg_metabolites': hvg_metabolites,
            'loading_matrix': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_top_hvg': n_top_hvg,
            'pca_dim': pca_dim
        }
        
        pca_file = output_dir / "metabolomics_pca_model.pkl"
        with open(pca_file, 'wb') as f:
            pickle.dump(pca_info, f)
        print(f"  💾 保存 PCA 模型: {pca_file}")
        
        # 保存 HVG 列表
        hvg_file = output_dir / "hvg_metabolites.txt"
        with open(hvg_file, 'w') as f:
            for metabolite in hvg_metabolites:
                f.write(f"{metabolite}\n")
        print(f"  💾 保存 HVG 列表: {hvg_file}")
    
    return adata_plasma_pca, adata_csf_pca, hvg_metabolites

def main():
    print("=" * 60)
    print("Step 0.3: 代谢组数据预处理 (修复版)")
    print("=" * 60)
    
    # ========== 1. 血浆代谢组 (ST000046) ==========
    print("\n【1/2】处理血浆代谢组 (ST000046)...")
    plasma_file = DATA_DIR / "st000046" / "ST000046_AN000076.txt"
    
    try:
        plasma_df, plasma_diagnosis = read_metabolomics_workbench(plasma_file)
        plasma_norm = preprocess_metabolomics(plasma_df, min_valid_ratio=0.5)
        
        # 转换为 AnnData（添加诊断信息）
        adata_plasma = ad.AnnData(
            X=plasma_norm.values,  # 样本 × 代谢物
            obs=plasma_diagnosis,  # 添加诊断信息
            var=pd.DataFrame(index=plasma_norm.columns)
        )
        adata_plasma.var['metabolite_id'] = adata_plasma.var.index
        
        # 保存
        output_plasma = OUTPUT_DIR / "plasma_metabolomics_common.h5ad"
        adata_plasma.write(output_plasma)
        print(f"💾 保存: {output_plasma}")
        print(f"   {adata_plasma.n_obs} 样本 × {adata_plasma.n_vars} 代谢物")
        
    except Exception as e:
        print(f"❌ 血浆代谢组处理失败: {e}")
        import traceback
        traceback.print_exc()
        adata_plasma = None
    
    # ========== 2. CSF 代谢组 (ST000047) ==========
    print("\n【2/2】处理 CSF 代谢组 (ST000047)...")
    csf_file = DATA_DIR / "st000047" / "ST000047_AN000080.txt"
    
    try:
        csf_df, csf_diagnosis = read_metabolomics_workbench(csf_file)
        csf_norm = preprocess_metabolomics(csf_df, min_valid_ratio=0.5)
        
        # 转换为 AnnData（添加诊断信息）
        adata_csf = ad.AnnData(
            X=csf_norm.values,
            obs=csf_diagnosis,  # 添加诊断信息
            var=pd.DataFrame(index=csf_norm.columns)
        )
        adata_csf.var['metabolite_id'] = adata_csf.var.index
        
        # 保存
        output_csf = OUTPUT_DIR / "csf_metabolomics_common.h5ad"
        adata_csf.write(output_csf)
        print(f"💾 保存: {output_csf}")
        print(f"   {adata_csf.n_obs} 样本 × {adata_csf.n_vars} 代谢物")
        
    except Exception as e:
        print(f"❌ CSF 代谢组处理失败: {e}")
        import traceback
        traceback.print_exc()
        adata_csf = None
    
    print("\n" + "=" * 60)
    print("✅ 代谢组预处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
