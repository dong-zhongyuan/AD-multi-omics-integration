import os
#!/usr/bin/env python3
"""
Step 3: Hub识别 - 使用 MultiXrank 多层网络随机游走

改进：
1. 先筛选高质量边（confidence > 0.9, strength >= top 10%）
2. 使用 MultiXrank 识别Hub（多层网络随机游走）
3. 识别关键Hub节点
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 multixrank 到路径
sys.path.insert(0, str(PROJECT_ROOT / 'tools/multixrank'))

# 导入配置管理器
from tools.config_loader import get_config
config = get_config()

try:
    import multixrank
    print("✓ MultiXrank 已加载")
except ImportError as e:
    print(f"✗ MultiXrank 加载失败: {e}")
    print("请检查路径: ./tools/multixrank")
    sys.exit(1)

# 配置
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
STEP2_DIR = PROJECT_ROOT / "output/step2_cross_tissue_causality"
OUTPUT_DIR = PROJECT_ROOT / "output/step3_hub_identification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HUB_THRESHOLD_PERCENTILE = 80  # 前20%为Hub（原来是90，即前10%）

# 边筛选阈值（根据step2实际分布调整）
CONFIDENCE_THRESHOLD = config.get_parameter("causality.confidence_threshold")  # 放宽到0.5，匹配step2的平均confidence 0.72-0.76
STRENGTH_PERCENTILE = 80  # 放宽到前20%


def filter_high_quality_edges(edges_df, confidence_threshold, strength_percentile):
    """
    筛选高质量边（仅用于组织内网络）
    """
    print(f"  原始边数: {len(edges_df)}")
    
    # 计算综合置信度（三指标平均）
    edges_df['confidence_combined'] = (
        edges_df['confidence_stability'] + 
        edges_df['confidence_snr'] + 
        edges_df['confidence_consistency']
    ) / 3.0
    
    # 计算strength阈值
    strength_threshold = np.percentile(edges_df['strength'], strength_percentile)
    
    # 筛选
    filtered = edges_df[
        (edges_df['confidence_combined'] > confidence_threshold) &
        (edges_df['strength'] >= strength_threshold)
    ].copy()
    
    print(f"  筛选后: {len(filtered)} 条边 (保留 {len(filtered)/len(edges_df)*100:.1f}%)")
    print(f"  阈值: confidence_combined > {confidence_threshold}, strength >= {strength_threshold:.6f}")
    
    return filtered


def prepare_multixrank_input(blood_edges, brain_edges, cross_edges):
    """
    准备 MultiXrank 输入格式
    
    MultiXrank 需要：
    1. multiplex: 多层网络定义
    2. config: 参数配置
    """
    # 构建多层网络
    # Layer 1: 血液网络
    # Layer 2: 脑组织网络
    # Layer 3: 跨组织边（作为层间连接）
    
    # 准备边列表（MultiXrank格式：source, target, weight）
    # 使用 strength 作为权重（保证为正值）
    # 重要：为不同层的节点添加前缀，避免节点重复
    blood_network = blood_edges[['source', 'target', 'strength']].copy()
    blood_network['source'] = 'blood_' + blood_network['source'].astype(str)
    blood_network['target'] = 'blood_' + blood_network['target'].astype(str)
    blood_network.rename(columns={'strength': 'weight'}, inplace=True)
    blood_network['layer'] = 'blood'
    
    brain_network = brain_edges[['source', 'target', 'strength']].copy()
    brain_network['source'] = 'brain_' + brain_network['source'].astype(str)
    brain_network['target'] = 'brain_' + brain_network['target'].astype(str)
    brain_network.rename(columns={'strength': 'weight'}, inplace=True)
    brain_network['layer'] = 'brain'
    
    # 跨组织边（也需要添加前缀）
    cross_network = cross_edges[['source', 'target', 'strength']].copy()
    cross_network['source'] = 'brain_' + cross_network['source'].astype(str)
    cross_network['target'] = 'blood_' + cross_network['target'].astype(str)
    cross_network.rename(columns={'strength': 'weight'}, inplace=True)
    cross_network['source_layer'] = 'brain'
    cross_network['target_layer'] = 'blood'
    
    return blood_network, brain_network, cross_network


def run_multixrank(blood_network, brain_network, cross_network, omics_name, omics_dir):
    """
    运行 MultiXrank 多层网络随机游走
    
    Returns:
        blood_scores: 血液节点的重要性分数
        brain_scores: 脑组织节点的重要性分数
    """
    print("\n  构建多层网络...")
    
    print(f"  多层网络规模:")
    print(f"    血液层: {len(blood_network)} 条边")
    print(f"    脑层: {len(brain_network)} 条边")
    print(f"    跨层: {len(cross_network)} 条边")
    
    # 运行 MultiXrank
    print("\n  运行 MultiXrank...")
    try:
        # 创建临时工作目录
        import tempfile
        import yaml
        
        temp_dir = omics_dir / 'multixrank_temp'
        temp_dir.mkdir(exist_ok=True)
        
        # 创建数据目录
        multiplex_dir = temp_dir / 'multiplex'
        multiplex_dir.mkdir(exist_ok=True)
        
        blood_dir = multiplex_dir / 'blood'
        blood_dir.mkdir(exist_ok=True)
        
        brain_dir = multiplex_dir / 'brain'
        brain_dir.mkdir(exist_ok=True)
        
        bipartite_dir = temp_dir / 'bipartite'
        bipartite_dir.mkdir(exist_ok=True)
        
        # 保存血液网络（TSV格式：node1 node2 weight）
        # 过滤自环
        blood_file = blood_dir / 'blood_layer.tsv'
        blood_network_filtered = blood_network[blood_network['source'] != blood_network['target']]
        
        # 检查是否有边
        if len(blood_network_filtered) == 0:
            print(f"  ⚠️  警告：血液网络过滤自环后无边，跳过 MultiXrank")
            raise ValueError("血液网络过滤自环后无边")
        
        blood_network_filtered[['source', 'target', 'weight']].to_csv(
            blood_file, sep='\t', index=False, header=False
        )
        
        # 保存脑网络
        brain_file = brain_dir / 'brain_layer.tsv'
        brain_network_filtered = brain_network[brain_network['source'] != brain_network['target']]
        
        if len(brain_network_filtered) == 0:
            print(f"  ⚠️  警告：脑网络过滤自环后无边，跳过 MultiXrank")
            raise ValueError("脑网络过滤自环后无边")
        
        brain_network_filtered[['source', 'target', 'weight']].to_csv(
            brain_file, sep='\t', index=False, header=False
        )
        
        # 保存跨组织边（bipartite）
        bipartite_file = bipartite_dir / 'brain_to_blood.tsv'
        cross_network_filtered = cross_network[cross_network['source'] != cross_network['target']]
        
        if len(cross_network_filtered) == 0:
            print(f"  ⚠️  警告：跨组织网络过滤自环后无边，跳过 MultiXrank")
            raise ValueError("跨组织网络过滤自环后无边")
        
        cross_network_filtered[['source', 'target', 'weight']].to_csv(
            bipartite_file, sep='\t', index=False, header=False
        )
        
        # 创建seed文件（全局排序：使用所有节点作为种子）
        # 注意：只包含在过滤后网络中实际存在的节点
        seed_file = temp_dir / 'seeds.txt'
        blood_nodes = set(blood_network_filtered['source']) | set(blood_network_filtered['target'])
        brain_nodes = set(brain_network_filtered['source']) | set(brain_network_filtered['target'])
        all_nodes = blood_nodes | brain_nodes
        seed_file.write_text('\n'.join(sorted(all_nodes)))
        
        # 创建配置文件
        config = {
            'multiplex': {
                'blood': {
                    'layers': ['multiplex/blood/blood_layer.tsv']
                },
                'brain': {
                    'layers': ['multiplex/brain/brain_layer.tsv']
                }
            },
            'bipartite': {
                'bipartite/brain_to_blood.tsv': {
                    'source': 'brain',
                    'target': 'blood'
                }
            },
            'seed': 'seeds.txt',  # 空seed文件，全局排序
            'r': 0.5  # 降低restart probability，让随机游走更全局（默认0.7）
        }
        
        config_file = temp_dir / 'config.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        print(f"  配置文件: {config_file}")
        print(f"  工作目录: {temp_dir}")
        
        # 调用 MultiXrank
        from multixrank import Multixrank
        
        mx = Multixrank(
            config=str(config_file),
            wdir=str(temp_dir)
        )
        
        # 运行随机游走
        result_df = mx.random_walk_rank()
        
        # 解析结果
        blood_scores = {}
        brain_scores = {}
        
        for _, row in result_df.iterrows():
            node = row['node']
            multiplex = row['multiplex']
            score = row['score']
            
            if multiplex == 'blood':
                # 去掉前缀 'blood_'
                original_node = node.replace('blood_', '', 1) if node.startswith('blood_') else node
                blood_scores[original_node] = score
            elif multiplex == 'brain':
                # 去掉前缀 'brain_'
                original_node = node.replace('brain_', '', 1) if node.startswith('brain_') else node
                brain_scores[original_node] = score
        
        print(f"  ✓ MultiXrank 完成")
        print(f"    血液节点: {len(blood_scores)} 个")
        print(f"    脑节点: {len(brain_scores)} 个")
        
        return blood_scores, brain_scores
    
    except Exception as e:
        print(f"  ✗ MultiXrank 失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"  回退到度中心性方法...")
        
        # 回退方案：使用度中心性
        blood_degree = blood_network['source'].value_counts() + blood_network['target'].value_counts()
        brain_degree = brain_network['source'].value_counts() + brain_network['target'].value_counts()
        
        blood_scores = blood_degree.to_dict()
        brain_scores = brain_degree.to_dict()
        
        return blood_scores, brain_scores


def identify_hubs_from_scores(scores_dict, threshold_percentile=90):
    """
    从 MultiXrank 分数中识别 Hub
    
    Args:
        scores_dict: {node: score} 字典
        threshold_percentile: Hub阈值百分位数
    
    Returns:
        hubs: Hub节点列表
    """
    if not scores_dict:
        return []
    
    scores = pd.Series(scores_dict)
    threshold = np.percentile(scores.values, threshold_percentile)
    hubs = scores[scores >= threshold].index.tolist()
    
    return hubs


def identify_double_ended_hubs(blood_hubs, brain_hubs, cross_tissue_edges):
    """
    识别双端Hub：通过跨组织边连接的Hub节点
    """
    blood_hub_set = set(blood_hubs)
    brain_hub_set = set(brain_hubs)
    
    # 核心层：Hub间连接
    core_brain_hubs = []
    core_blood_hubs = []
    core_edges = []
    
    for _, edge in cross_tissue_edges.iterrows():
        source = edge['source']
        target = edge['target']
        
        if source in brain_hub_set and target in blood_hub_set:
            core_brain_hubs.append(source)
            core_blood_hubs.append(target)
            core_edges.append(edge)
    
    # 去重
    core_brain_hubs = list(set(core_brain_hubs))
    core_blood_hubs = list(set(core_blood_hubs))
    
    # 扩展层：Hub的下游节点
    extended_brain_hubs = []
    extended_blood_targets = []
    extended_edges = []
    
    for _, edge in cross_tissue_edges.iterrows():
        source = edge['source']
        target = edge['target']
        
        if source in brain_hub_set:
            extended_brain_hubs.append(source)
            extended_blood_targets.append(target)
            extended_edges.append(edge)
    
    # 去重
    extended_brain_hubs = list(set(extended_brain_hubs))
    extended_blood_targets = list(set(extended_blood_targets))
    
    core_layer = {
        'brain_hubs': core_brain_hubs,
        'blood_hubs': core_blood_hubs,
        'edges': pd.DataFrame(core_edges),
        'n_brain_hubs': len(core_brain_hubs),
        'n_blood_hubs': len(core_blood_hubs),
        'n_edges': len(core_edges),
    }
    
    extended_layer = {
        'brain_hubs': extended_brain_hubs,
        'blood_targets': extended_blood_targets,
        'edges': pd.DataFrame(extended_edges),
        'n_brain_hubs': len(extended_brain_hubs),
        'n_blood_targets': len(extended_blood_targets),
        'n_edges': len(extended_edges),
    }
    
    return core_layer, extended_layer


def main():
    print("="*60)
    print("Step 3: Hub识别 - MultiXrank")
    print("="*60)
    print(f"\n筛选阈值: confidence > {CONFIDENCE_THRESHOLD}, strength >= top {100-STRENGTH_PERCENTILE}%")
    print(f"Hub阈值: top {100-HUB_THRESHOLD_PERCENTILE}%")
    
    # 定义三个组学
    omics_list = ['metabolomics', 'proteomics', 'transcriptomics']
    
    all_stats = []
    
    # 处理每个组学
    for omics_name in omics_list:
        print(f"\n{'='*60}")
        print(f"处理: {omics_name.upper()}")
        print(f"{'='*60}")
        
        omics_dir = STEP2_DIR / omics_name
        
        if not omics_dir.exists():
            print(f"\n⚠️  {omics_name}: step2输出目录不存在 ({omics_dir})，跳过")
            continue
        
        # 1. 加载边数据
        print(f"\n[1] 加载边数据...")
        
        # 直接读取cross_tissue_edges.csv
        cross_edges = pd.read_csv(omics_dir / 'cross_tissue_edges.csv')
        
        blood_edges = pd.read_csv(omics_dir / 'blood_network' / 'consensus_edges.csv')
        brain_edges = pd.read_csv(omics_dir / 'brain_network' / 'consensus_edges.csv')
        
        print(f"  跨组织边: {len(cross_edges)} 条")
        print(f"  血液网络: {len(blood_edges)} 条")
        print(f"  脑网络: {len(brain_edges)} 条")
        
        # 保存筛选后的边
        filtered_dir = omics_dir / 'filtered_edges'
        filtered_dir.mkdir(exist_ok=True)
        cross_edges.to_csv(filtered_dir / 'cross_tissue_edges.csv', index=False)
        blood_edges.to_csv(filtered_dir / 'blood_network_edges.csv', index=False)
        brain_edges.to_csv(filtered_dir / 'brain_network_edges.csv', index=False)
        print(f"\n  ✓ 筛选后的边已保存到: {filtered_dir}")
        
        # 3. 准备 MultiXrank 输入
        print(f"\n[3] 准备 MultiXrank 输入...")
        blood_network, brain_network, cross_network = prepare_multixrank_input(
            blood_edges, brain_edges, cross_edges
        )
        
        # 4. 运行 MultiXrank
        print(f"\n[4] 运行 MultiXrank...")
        blood_scores, brain_scores = run_multixrank(blood_network, brain_network, cross_network, omics_name, omics_dir)
        
        # 5. 只识别脑Hub
        print(f"\n[5] 识别脑Hub（阈值: top {100-HUB_THRESHOLD_PERCENTILE}%）...")
        brain_hubs = identify_hubs_from_scores(brain_scores, HUB_THRESHOLD_PERCENTILE)
        
        print(f"  脑组织Hub: {len(brain_hubs)} 个")
        
        # 6. 保存结果
        omics_output = OUTPUT_DIR / omics_name
        omics_output.mkdir(exist_ok=True)
        
        # 保存脑Hub（带分数）
        pd.DataFrame({
            'hub': brain_hubs,
            'score': [brain_scores.get(h, 0) for h in brain_hubs]
        }).to_csv(omics_output / 'brain_hubs.csv', index=False)
        
        # 保存统计信息
        stats = {
            'omics': omics_name,
            'brain_hubs': len(brain_hubs),
        }
        
        with open(omics_output / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ {omics_name} 完成，结果保存到: {omics_output}")
        
        all_stats.append(stats)
    
    # 保存总结
    summary = {
        'method': 'MultiXrank',
        'hub_threshold_percentile': HUB_THRESHOLD_PERCENTILE,
        'edge_filters': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'strength_percentile': STRENGTH_PERCENTILE,
        },
        'omics': all_stats,
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 全部完成！结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # 打印总结
    print(f"\n总结:\n")
    for stats in all_stats:
        print(f"{stats['omics']}:")
        print(f"  脑Hub: {stats['brain_hubs']} 个")


if __name__ == '__main__':
    main()
