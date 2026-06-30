#!/usr/bin/env python3
"""Figure 1a - Study workflow, BioRender-style.

A vertical pipeline of five colored stage cards, each with a circular icon
badge, a bold stage title, and a one-line caption. Cards connected by chevron
arrows in the stage accent color. Replaces the earlier plain-hollow-box
schematic with a richer, publication-figure "graphical abstract" look while
keeping the same 10x10 panel size and the project color palette.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── pipeline stages ────────────────────────────────────────────────────
# each: accent color, icon glyph, title, caption
stages = [
    {"color": "#5B8FF9", "icon": "🧬", "title": "Multi-Omics Data",
     "sub": "Brain · Blood · CSF  (3 layers)"},
    {"color": "#7B61FF", "icon": "⚙", "title": "Neural ODE + OT Mapping",
     "sub": "Continuous cross-tissue vector field"},
    {"color": "#F6BD16", "icon": "🕸", "title": "Jacobian Sensitivity Networks",
     "sub": "Brain → Blood causal edges"},
    {"color": "#F08BB4", "icon": "✂", "title": "Virtual Knockout Validation",
     "sub": "GenKI + Geneformer, 22 edges"},
    {"color": "#5AD8A6", "icon": "🏥", "title": "Clinical Translation",
     "sub": "Diagnosis · Prognosis · Targets"},
]

n = len(stages)
fig_w, fig_h = 10, 10
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis('off')

# layout
card_w = 7.4
card_h = 1.30
gap = 0.42
total_h = n * card_h + (n - 1) * gap
y_top = (fig_h + total_h) / 2 - card_h / 2

card_left = (fig_w - card_w) / 2
cx = fig_w / 2

for i, st in enumerate(stages):
    cy = y_top - i * (card_h + gap)
    color = st["color"]

    # soft drop shadow
    shadow = FancyBboxPatch(
        (card_left + 0.06, cy - card_h / 2 - 0.06), card_w, card_h,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor="#000000", alpha=0.10, linewidth=0, zorder=1)
    ax.add_patch(shadow)

    # card body: white fill, accent left rail, thin border
    card = FancyBboxPatch(
        (card_left, cy - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor='white', edgecolor=color, linewidth=2.4, zorder=2)
    ax.add_patch(card)
    # colored left rail (accent strip)
    rail = FancyBboxPatch(
        (card_left, cy - card_h / 2), 0.22, card_h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        facecolor=color, edgecolor='none', linewidth=0, zorder=3)
    ax.add_patch(rail)

    # circular icon badge
    badge_r = 0.40
    badge_x = card_left + 0.22 + badge_r + 0.22
    badge = Circle((badge_x, cy), badge_r, facecolor=color, edgecolor='white',
                   linewidth=2.5, zorder=4)
    ax.add_patch(badge)
    ax.text(badge_x, cy, st["icon"], ha='center', va='center',
            fontsize=30, zorder=5)

    # title + caption to the right of the badge
    text_x = badge_x + badge_r + 0.30
    ax.text(text_x, cy + 0.22, st["title"], ha='left', va='center',
            fontsize=26, fontweight='bold', color='#1a1a1a', zorder=5)
    ax.text(text_x, cy - 0.26, st["sub"], ha='left', va='center',
            fontsize=17, color='#555555', zorder=5)

    # connecting arrow to next card
    if i < n - 1:
        y0 = cy - card_h / 2 - 0.04
        y1 = cy - card_h / 2 - gap + 0.04
        arr = FancyArrowPatch(
            (cx, y0), (cx, y1),
            arrowstyle='-|>,head_length=8,head_width=6',
            linewidth=3.0, color=color, zorder=2,
            mutation_scale=1.0)
        ax.add_patch(arr)

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
save(fig, OUT_DIR, 'fig1a')
