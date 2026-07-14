import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8, 'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12
})

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_file = os.path.join(base_dir, 'data', 'fig1b_umap.csv')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(in_file)
    
    # 随机下采样至最多2万个点以保证SVG渲染性能和大小
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
        
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    
    # 颜色映射字典
    color_tissue = {'Brain': '#0072B2', 'Blood': '#C44E52'}
    color_geno = {'WT': '#999999', '5xFAD': '#E69F00'}
    
    # Left Panel: Tissue
    ax = axes[0]
    for tissue, group in df.groupby('tissue'):
        ax.scatter(group['UMAP1'], group['UMAP2'], s=2, alpha=0.6, 
                   color=color_tissue.get(tissue, '#333333'), label=tissue, edgecolors='none')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(frameon=False, markerscale=4, loc='best')
    
    # Right Panel: Predicted Genotype
    ax = axes[1]
    for geno, group in df.groupby('predicted_genotype'):
        ax.scatter(group['UMAP1'], group['UMAP2'], s=2, alpha=0.6, 
                   color=color_geno.get(geno, '#333333'), label=geno, edgecolors='none')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(frameon=False, markerscale=4, loc='best')
    
    # 移除顶部和右侧脊线，隐藏刻度数字（UMAP无需刻度值）
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(out_dir, f'b.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()