import os
#!/usr/bin/env python3
"""
可视化Step3筛选后的边的各项指标趋势
使用折线图展示按rank排序的指标变化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加项目根目录
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_loader import get_config
config = get_config()

# 设置绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 路径配置
STEP3_DIR = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis"
OUTPUT_DIR = PROJECT_ROOT / "output/step3_hub_identification/eigengene_analysis"

OMICS_TYPES = ['proteomics', 'transcriptomics', 'metabolomics']


def visualize_edge_metrics(omics_name):
    """
    可视化单个组学的边指标趋势（折线图）
    """
    print(f"\n{'='*60}")
    print(f"可视化: {omics_name}")
    print(f"{'='*60}")
    
    # 加载筛选后的边数据（包含完整的评分指标）
    edges_file = STEP3_DIR / omics_name / "filtered_cross_tissue_edges.csv"
    if not edges_file.exists():
        print(f"  ⚠️  边文件不存在: {edges_file}")
        return
    
    edges_df = pd.read_csv(edges_file)
    print(f"  加载 {len(edges_df)} 条边（筛选后）")
    
    if len(edges_df) == 0:
        print(f"  ⚠️  无边数据，跳过可视化")
        return
    
    # 如果没有rank列，按final_score降序排序生成rank
    if 'rank' not in edges_df.columns:
        if 'final_score' in edges_df.columns:
            edges_df = edges_df.sort_values('final_score', ascending=False).reset_index(drop=True)
            edges_df['rank'] = edges_df.index + 1
        else:
            print(f"  ⚠️  缺少rank和final_score列，无法排序")
            return
    else:
        edges_df = edges_df.sort_values('rank').reset_index(drop=True)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 原始置信度指标趋势（左上）
    ax1 = plt.subplot(2, 3, 1)
    if all(m in edges_df.columns for m in ['confidence_stability', 'confidence_snr', 'confidence_consistency']):
        ax1.plot(edges_df['rank'], edges_df['confidence_stability'], 
                label='Stability', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax1.plot(edges_df['rank'], edges_df['confidence_snr'], 
                label='SNR', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax1.plot(edges_df['rank'], edges_df['confidence_consistency'], 
                label='Consistency', linewidth=2, marker='^', markersize=3, alpha=0.7)
        ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax1.set_title('Raw Confidence Metrics Trend', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # 2. 归一化置信度指标趋势（中上）
    ax2 = plt.subplot(2, 3, 2)
    if all(m in edges_df.columns for m in ['stability_norm', 'snr_norm', 'consistency_norm']):
        ax2.plot(edges_df['rank'], edges_df['stability_norm'], 
                label='Stability (norm)', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax2.plot(edges_df['rank'], edges_df['snr_norm'], 
                label='SNR (norm)', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax2.plot(edges_df['rank'], edges_df['consistency_norm'], 
                label='Consistency (norm)', linewidth=2, marker='^', markersize=3, alpha=0.7)
        ax2.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
        ax2.set_title('Normalized Confidence Metrics Trend', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # 3. Strength趋势（右上）
    ax3 = plt.subplot(2, 3, 3)
    if 'strength' in edges_df.columns:
        ax3.plot(edges_df['rank'], edges_df['strength'], 
                linewidth=2.5, color='darkred', marker='o', markersize=3, alpha=0.8)
        ax3.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Strength', fontsize=12, fontweight='bold')
        ax3.set_title('Edge Strength Trend', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加中位数线
        median_strength = edges_df['strength'].median()
        ax3.axhline(median_strength, color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: {median_strength:.3f}')
        ax3.legend(loc='best', fontsize=10)
    
    # 4. 综合评分趋势（左下）
    ax4 = plt.subplot(2, 3, 4)
    if all(m in edges_df.columns for m in ['confidence_combined', 'confidence_ko_optimized']):
        ax4.plot(edges_df['rank'], edges_df['confidence_combined'], 
                label='Combined (3-metric avg)', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax4.plot(edges_df['rank'], edges_df['confidence_ko_optimized'], 
                label='KO-optimized (snr+stability)', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax4.set_title('Combined Confidence Metrics Trend', fontsize=13, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    # 5. Final Score趋势（中下）
    ax5 = plt.subplot(2, 3, 5)
    if 'final_score' in edges_df.columns:
        ax5.plot(edges_df['rank'], edges_df['final_score'], 
                linewidth=2.5, color='darkgreen', marker='o', markersize=4, alpha=0.8)
        ax5.set_xlabel('Rank', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Final Score', fontsize=12, fontweight='bold')
        ax5.set_title('Final Score Trend (strength × confidence)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 标注骤降点（如果有elbow_drop_analysis.csv）
        elbow_file = STEP3_DIR / omics_name / "elbow_drop_analysis.csv"
        if elbow_file.exists():
            elbow_df = pd.read_csv(elbow_file)
            if len(elbow_df) > 1:
                max_drop_idx = elbow_df['drop'].idxmax()
                elbow_rank = elbow_df.loc[max_drop_idx, 'start_rank']
                ax5.axvline(elbow_rank, color='red', linestyle='--', linewidth=2, 
                           label=f'Elbow point: rank {int(elbow_rank)}')
                ax5.legend(loc='best', fontsize=10)
    
    # 6. 统计摘要（右下）
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 计算统计信息
    stats_text = f"{'='*40}\n"
    stats_text += f"{omics_name.upper()} Statistics\n"
    stats_text += f"{'='*40}\n\n"
    stats_text += f"Total Edges (after filtering): {len(edges_df)}\n\n"
    
    if 'final_score' in edges_df.columns:
        fs = edges_df['final_score'].dropna()
        stats_text += f"Final Score:\n"
        stats_text += f"  Mean:   {fs.mean():.4f}\n"
        stats_text += f"  Median: {fs.median():.4f}\n"
        stats_text += f"  Std:    {fs.std():.4f}\n"
        stats_text += f"  Range:  [{fs.min():.4f}, {fs.max():.4f}]\n\n"
    
    if 'strength' in edges_df.columns:
        st = edges_df['strength'].dropna()
        stats_text += f"Strength:\n"
        stats_text += f"  Mean:   {st.mean():.4f}\n"
        stats_text += f"  Median: {st.median():.4f}\n"
        stats_text += f"  Range:  [{st.min():.4f}, {st.max():.4f}]\n\n"
    
    if 'confidence_combined' in edges_df.columns:
        cc = edges_df['confidence_combined'].dropna()
        stats_text += f"Confidence Combined:\n"
        stats_text += f"  Mean:   {cc.mean():.4f}\n"
        stats_text += f"  Median: {cc.median():.4f}\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 总标题
    fig.suptitle(f'{omics_name.upper()} - Edge Metrics Trend (After Filtering)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存
    output_file = OUTPUT_DIR / omics_name / f"{omics_name}_edge_metrics_trend.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Step3 边指标趋势可视化（筛选后）")
    print("="*60)
    
    for omics in OMICS_TYPES:
        try:
            visualize_edge_metrics(omics)
        except Exception as e:
            print(f"  ✗ {omics} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ 全部完成！")
    print("="*60)


if __name__ == "__main__":
    main()
