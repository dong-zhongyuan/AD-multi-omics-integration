import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8, 'axes.labelsize': 14, 'axes.labelweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11
})

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_file = os.path.join(base_dir, 'data', 'fig1c_training.csv')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(in_file)
    omics_layers = df['omics'].unique()
    
    fig, axes = plt.subplots(1, len(omics_layers), figsize=(10, 3), constrained_layout=True)
    if len(omics_layers) == 1:
        axes = [axes]
        
    for ax, omics in zip(axes, omics_layers):
        sub_df = df[df['omics'] == omics]
        
        ax.plot(sub_df['epoch'], sub_df['train_total'], color='#0072B2', linewidth=2, linestyle='-', label='Train')
        ax.plot(sub_df['epoch'], sub_df['val_total'], color='#D55E00', linewidth=2, linestyle='--', label='Val')
        
        ax.set_xlabel('Epoch')
        if ax == axes[0]:
            ax.set_ylabel('Total Loss')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # 只在最右侧子图放Legend
    axes[-1].legend(frameon=False, loc='upper right')
        
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(out_dir, f'c.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()