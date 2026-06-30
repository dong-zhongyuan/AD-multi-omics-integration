#!/usr/bin/env python3
"""Plot: fig6c_processed.csv → fig6c.svg/pdf/tiff
Gene Classification — stacked/grouped view showing Therapeutic vs Diagnostic assignment.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import *
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT_DIR, 'fig6c_processed.csv'))

flow = df.groupby(['omics_label', 'final_class']).size().reset_index(name='count')

left_order = ['Transcriptomics', 'Proteomics']
right_order = ['Diagnostic', 'Both', 'Therapeutic']
left_colors = {
    'Transcriptomics': C_TRANSCRIPTOMICS,
    'Proteomics': C_PROTEOMICS,
}
right_colors = {
    'Diagnostic': C_BLUE,
    'Both': C_PURPLE,
    'Therapeutic': C_RED,
}

def ribbon(ax, x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color):
    curve = (x1 - x0) * 0.42
    verts = [
        (x0, y0_top),
        (x0 + curve, y0_top),
        (x1 - curve, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        (x1 - curve, y1_bottom),
        (x0 + curve, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor=color,
                      lw=1.6, alpha=0.22)
    ax.add_patch(patch)

def stack_positions(totals, order, gap, scale, y_min=0.10, y_max=0.90):
    total_h = sum(totals.get(name, 0) * scale for name in order) + gap * (len(order) - 1)
    top = y_min + (y_max - y_min + total_h) / 2
    positions = {}
    cursor = top
    for name in order:
        h = totals.get(name, 0) * scale
        positions[name] = (cursor - h, cursor)
        cursor = cursor - h - gap
    return positions

left_totals = flow.groupby('omics_label')['count'].sum().to_dict()
right_totals = flow.groupby('final_class')['count'].sum().to_dict()
total_n = int(flow['count'].sum())
left_gap = 0.12
right_gap = 0.08
available_left = 0.80 - left_gap * (len(left_order) - 1)
available_right = 0.80 - right_gap * (len(right_order) - 1)
scale = min(available_left, available_right) / max(total_n, 1)

left_pos = stack_positions(left_totals, left_order, left_gap, scale)
right_pos = stack_positions(right_totals, right_order, right_gap, scale)

left_intervals = {}
left_cursor = {name: left_pos[name][1] for name in left_order}
for omics_label in left_order:
    for final_class in right_order:
        mask = (flow['omics_label'] == omics_label) & (flow['final_class'] == final_class)
        count = int(flow.loc[mask, 'count'].sum())
        if count == 0:
            continue
        height = count * scale
        top = left_cursor[omics_label]
        bottom = top - height
        left_intervals[(omics_label, final_class)] = (bottom, top)
        left_cursor[omics_label] = bottom

right_intervals = {}
right_cursor = {name: right_pos[name][1] for name in right_order}
for final_class in right_order:
    for omics_label in left_order:
        mask = (flow['omics_label'] == omics_label) & (flow['final_class'] == final_class)
        count = int(flow.loc[mask, 'count'].sum())
        if count == 0:
            continue
        height = count * scale
        top = right_cursor[final_class]
        bottom = top - height
        right_intervals[(omics_label, final_class)] = (bottom, top)
        right_cursor[final_class] = bottom

fig, ax = plt.subplots(figsize=(30, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# node positions tuned for the 3:1 canvas: spread the two columns wide so the
# ribbons span the panel without dead margins
left_x = 0.10
right_x = 0.80
node_w = 0.05

for (omics_label, final_class), (y0_bottom, y0_top) in left_intervals.items():
    y1_bottom, y1_top = right_intervals[(omics_label, final_class)]
    ribbon(ax, left_x + node_w, right_x, y0_bottom, y0_top, y1_bottom, y1_top, left_colors[omics_label])

for name in left_order:
    y0, y1 = left_pos[name]
    ax.add_patch(Rectangle((left_x, y0), node_w, y1 - y0, facecolor='white',
                           edgecolor=left_colors[name], linewidth=2.5))
    ax.text(left_x - 0.03, (y0 + y1) / 2, f'{name} (n={left_totals.get(name, 0)})',
            ha='right', va='center', fontsize=F_VALUE, fontweight='bold')

for name in right_order:
    y0, y1 = right_pos[name]
    ax.add_patch(Rectangle((right_x, y0), node_w, y1 - y0, facecolor='white',
                           edgecolor=right_colors[name], linewidth=2.5))
    ax.text(right_x + node_w + 0.03, (y0 + y1) / 2, f'{name} (n={right_totals.get(name, 0)})',
            ha='left', va='center', fontsize=F_VALUE, fontweight='bold')

ax.text(left_x + node_w / 2, 0.95, 'Omics layer', ha='center', va='bottom',
        fontsize=F_VALUE, fontweight='bold')
ax.text(right_x + node_w / 2, 0.95, 'Candidate role', ha='center', va='bottom',
        fontsize=F_VALUE, fontweight='bold')
fig.subplots_adjust(left=0.14, right=0.95, top=0.93, bottom=0.16)
save(fig, OUT_DIR, 'fig6c')
