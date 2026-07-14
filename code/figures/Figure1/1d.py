import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8, 'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12
})

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_file = os.path.join(base_dir, 'data', 'fig1d_benchmark.csv')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(in_file)
    # 按OT distance排序以获得更好的视觉递增/递减效果，或保持原始顺序。这里保持原始顺序。
    methods = df['method'].tolist()
    
    colors = {'NeuralODE': '#0072B2', 'Ridge': '#E69F00', 'DirectOT': '#999999', 'Identity': '#CC79A7'}
    
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    
    for i, row in df.iterrows():
        m = row['method']
        color = colors.get(m, '#333333')
        
        # 绘制 Error bar 和 Dot
        ax.errorbar(i, row['ot_distance_mean'], yerr=row['ot_distance_sd'], 
                    fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5, markersize=8)
        
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('OT Distance')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(out_dir, f'd.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()