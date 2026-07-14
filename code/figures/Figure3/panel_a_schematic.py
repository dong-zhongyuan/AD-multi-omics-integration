# panel_a_schematic.py — Figure 3: Dual-method VK validation workflow
import os, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'axes.linewidth':0.8})
C_GENKI='#0072B2'; C_PPI='#D55E00'; C_SCENIC='#009E73'

def render_cards(stages, out_dir):
    fig_w, fig_h = 3.0, 3.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.set_xlim(0,fig_w); ax.set_ylim(0,fig_h); ax.axis('off')
    n=len(stages); mx=0.04; mt=0.04; mb=0.04; gap=0.08
    cw=fig_w-2*mx; ch=(fig_h-mt-mb-gap*(n-1))/n; RAD=0.06; cl=mx
    def cy(i): return fig_h-mt-(i+1)*ch-i*gap
    for i,st in enumerate(stages):
        y0=cy(i); c=y0+ch/2; color=st['color']
        ax.add_patch(FancyBboxPatch((cl+0.04,y0-0.04),cw,ch,boxstyle=f'round,pad=0.02,rounding_size={RAD}',facecolor='#000000',alpha=0.07,linewidth=0,zorder=1))
        cs=FancyBboxPatch((cl,y0),cw,ch,boxstyle=f'round,pad=0.02,rounding_size={RAD}',facecolor=color,edgecolor='none',linewidth=0,zorder=2); ax.add_patch(cs)
        rw=0.14; wf=Rectangle((cl+rw,y0-0.06),cw-rw+0.12,ch+0.12,facecolor='white',edgecolor='none',linewidth=0,zorder=3); wf.set_clip_path(cs); ax.add_patch(wf)
        ax.add_patch(FancyBboxPatch((cl,y0),cw,ch,boxstyle=f'round,pad=0.02,rounding_size={RAD}',facecolor='none',edgecolor=color,linewidth=1.5,zorder=4))
        br=0.16; bx=cl+rw+br+0.05; ax.add_patch(Circle((bx,c),br,facecolor=color,edgecolor='white',linewidth=1.5,zorder=6))
        ax.text(bx,c,st['badge'],ha='center',va='center',fontsize=13,fontweight='bold',color='white',zorder=7)
        tx=bx+br+0.08; ax.text(tx,c+0.12,st['title'],ha='left',va='center',fontsize=13,fontweight='bold',color='#1A1A1A',zorder=5)
        ax.text(tx,c-0.14,st['data'],ha='left',va='center',fontsize=8,color='#6B7280',zorder=5)
        if i<n-1:
            cx2=fig_w/2; ax.add_patch(FancyArrowPatch((cx2,y0-0.01),(cx2,cy(i+1)+ch+0.01),arrowstyle='-|>',mutation_scale=12,color='#9CA3AF',linewidth=1.5,zorder=2))
    os.makedirs(out_dir,exist_ok=True)
    for ext in ['png','pdf','svg']: plt.savefig(os.path.join(out_dir,f'a.{ext}'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close()

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'..','output')
render_cards([
    {'color':C_GENKI,'badge':'1','title':'GenKI VK','data':'VGAE latent shift'},
    {'color':C_PPI,'badge':'2','title':'PPI Propagation','data':'Tissue-specific RWR'},
    {'color':C_SCENIC,'badge':'3','title':'SCENIC GRN','data':'Regulon activity'},
    {'color':C_GENKI,'badge':'4','title':'Concordance','data':'Cross-method validation'},
], OUT)
print('Saved a.png/pdf/svg')
