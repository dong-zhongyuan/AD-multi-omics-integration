# panel_a_schematic.py — Figure 5: Drug repurposing workflow (vertical card stack).
# Mirrors Figure 1/2/3/4 panel a aesthetic.
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'font.size': 17,
})

C_VK = '#0072B2'; C_SURV = '#009E73'; C_DRUG = '#D55E00'; C_TIER = '#444444'


def render_cards(stages, out_dir):
    fig_w, fig_h = 3.0, 3.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')
    n = len(stages)
    margin_x, margin_top, margin_bot, gap = 0.04, 0.04, 0.04, 0.08
    card_w = fig_w - 2 * margin_x
    card_h = (fig_h - margin_top - margin_bot - gap * (n - 1)) / n
    RAD = 0.06; card_left = margin_x
    def card_y(i):
        return fig_h - margin_top - (i + 1) * card_h - i * gap
    for i, st in enumerate(stages):
        y0 = card_y(i); cy = y0 + card_h / 2; color = st['color']
        ax.add_patch(FancyBboxPatch((card_left + 0.04, y0 - 0.04), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}', facecolor='#000000', alpha=0.07, linewidth=0, zorder=1))
        card_shape = FancyBboxPatch((card_left, y0), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}', facecolor=color, edgecolor='none', linewidth=0, zorder=2)
        ax.add_patch(card_shape)
        rail_w = 0.14
        wf = Rectangle((card_left + rail_w, y0 - 0.06), card_w - rail_w + 0.12, card_h + 0.12,
            facecolor='white', edgecolor='none', linewidth=0, zorder=3)
        wf.set_clip_path(card_shape); ax.add_patch(wf)
        ax.add_patch(FancyBboxPatch((card_left, y0), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}', facecolor='none', edgecolor=color, linewidth=1.6, zorder=4))
        badge_r = 0.16; badge_x = card_left + rail_w + badge_r + 0.05
        ax.add_patch(Circle((badge_x, cy), badge_r, facecolor=color, edgecolor='white', linewidth=1.5, zorder=6))
        ax.text(badge_x, cy, st['badge'], ha='center', va='center', fontsize=13, fontweight='bold', color='white', zorder=7)
        text_x = badge_x + badge_r + 0.08
        ax.text(text_x, cy + 0.12, st['title'], ha='left', va='center', fontsize=11, fontweight='bold', color='#1A1A1A', zorder=5)
        ax.text(text_x, cy - 0.14, st['data'], ha='left', va='center', fontsize=7, color='#6B7280', zorder=5)
        if i < n - 1:
            cx = fig_w / 2
            ax.add_patch(FancyArrowPatch((cx, y0 - 0.01), (cx, card_y(i + 1) + card_h + 0.01),
                arrowstyle='-|>', mutation_scale=12, color='#9CA3AF', linewidth=1.5, zorder=2))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(out_dir, f'a.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')
render_cards([
    {'color': C_VK,   'badge': '1', 'title': 'Reverse VK Targets',  'data': 'Blood-end KO genes'},
    {'color': C_SURV, 'badge': '2', 'title': 'Survival Filter',      'data': 'Cox p < 0.05'},
    {'color': C_DRUG, 'badge': '3', 'title': 'Drug Mining',          'data': 'OpenTargets + DGIdb'},
    {'color': C_TIER, 'badge': '4', 'title': 'Tiered Ranking',       'data': 'Phase 3 / 2 / Tractable'},
], OUT)
print('Saved a.png/pdf/svg')
