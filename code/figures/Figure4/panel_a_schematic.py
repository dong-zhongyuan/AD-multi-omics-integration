# panel_a_schematic.py — Figure 4: Diagnostic validation workflow (vertical card stack).
# Mirrors Figure 1/2/3 panel a aesthetic: cards stacked vertically, colored left rail,
# numbered badge, title + subtitle, downward arrows between. Square 1:1 canvas.
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.8,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'font.size': 17,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

C_POOL   = '#0072B2'
C_COHORT = '#D55E00'
C_EVAL   = '#009E73'
C_VALID  = '#444444'


def render_cards(stages, out_dir):
    fig_w, fig_h = 3.0, 3.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')

    n = len(stages)
    margin_x = 0.04
    margin_top = 0.04
    margin_bot = 0.04
    gap = 0.08
    card_w = fig_w - 2 * margin_x
    card_h = (fig_h - margin_top - margin_bot - gap * (n - 1)) / n
    RAD = 0.06
    card_left = margin_x

    def card_y(i):
        y_top = fig_h - margin_top
        return y_top - (i + 1) * card_h - i * gap

    for i, st in enumerate(stages):
        y0 = card_y(i)
        cy = y0 + card_h / 2
        color = st['color']

        # Drop shadow
        ax.add_patch(FancyBboxPatch(
            (card_left + 0.04, y0 - 0.04), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}',
            facecolor='#000000', alpha=0.07, linewidth=0, zorder=1))

        card_shape = FancyBboxPatch(
            (card_left, y0), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}',
            facecolor=color, edgecolor='none', linewidth=0, zorder=2)
        ax.add_patch(card_shape)

        # White inner fill (colored left rail)
        rail_w = 0.14
        white_fill = Rectangle(
            (card_left + rail_w, y0 - 0.06),
            card_w - rail_w + 0.12, card_h + 0.12,
            facecolor='white', edgecolor='none', linewidth=0, zorder=3)
        white_fill.set_clip_path(card_shape)
        ax.add_patch(white_fill)

        # Colored outline
        ax.add_patch(FancyBboxPatch(
            (card_left, y0), card_w, card_h,
            boxstyle=f'round,pad=0.02,rounding_size={RAD}',
            facecolor='none', edgecolor=color, linewidth=1.6, zorder=4))

        # Numbered badge
        badge_r = 0.16
        badge_x = card_left + rail_w + badge_r + 0.05
        ax.add_patch(Circle((badge_x, cy), badge_r, facecolor=color,
                            edgecolor='white', linewidth=1.5, zorder=6))
        ax.text(badge_x, cy, st['badge'], ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=7)

        # Title (bold, black) + subtitle (small gray)
        text_x = badge_x + badge_r + 0.08
        ax.text(text_x, cy + 0.12, st['title'], ha='left', va='center',
                fontsize=11, fontweight='bold', color='#1A1A1A', zorder=5)
        ax.text(text_x, cy - 0.14, st['data'], ha='left', va='center',
                fontsize=7, color='#6B7280', zorder=5)

        # Downward arrow to next card
        if i < n - 1:
            cx = fig_w / 2
            ax.add_patch(FancyArrowPatch(
                (cx, y0 - 0.01),
                (cx, card_y(i + 1) + card_h + 0.01),
                arrowstyle='-|>', mutation_scale=12,
                color='#9CA3AF', linewidth=1.5, zorder=2))

    os.makedirs(out_dir, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(out_dir, f'a.{ext}'), dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close()


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')

render_cards([
    {'color': C_POOL,   'badge': '1', 'title': 'VK Biomarker Pool',  'data': 'Proteomics forward targets'},
    {'color': C_COHORT, 'badge': '2', 'title': 'ADNI Cohort',         'data': 'CN / MCI / AD staging'},
    {'color': C_EVAL,   'badge': '3', 'title': 'AUC + Cox Survival',  'data': 'Diagnostic + prognostic'},
    {'color': C_VALID,  'badge': '4', 'title': 'MAPT Validated',      'data': 'AUC 0.719, HR 1.14'},
], OUT)
print('Saved a.png/pdf/svg')
