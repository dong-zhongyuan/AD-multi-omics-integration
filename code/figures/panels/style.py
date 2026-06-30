#!/usr/bin/env python3
"""Shared style module for AD Multi-Omics Integration figures.

All plot scripts import from this module:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from style import *
    apply_style()

DESIGN: svgpic + natbarplot standard.
- Large bold fonts (base 35pt, label 37pt, tick 32pt)
- Thick spines (2.5) and ticks (length 8, width 2.5)
- Hollow bars with colored edges, no fill
- Different marker shapes per group, hollow
- No in-figure titles, no panel labels (added in PPT/Illustrator)
- No annotation boxes overlaying data values
- No gradient fills, glow effects, or shadows
- Output: individual SVG + PDF for manual composition
"""
import os
import hashlib
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# COLOR CONSTANTS (Okabe-Ito + Multi-Omics Convention)
# ═══════════════════════════════════════════════════════════════

# Omics layers — primary
C_TRANSCRIPTOMICS = '#0072B2'   # deep blue
C_PROTEOMICS      = '#D55E00'   # vermillion
C_METABOLOMICS    = '#009E73'   # green

# Tissue
C_BRAIN           = '#7B2D8B'   # rich purple (CNS)
C_BLOOD           = '#E8590C'   # warm orange (periphery)

# Disease
C_AD              = '#C62828'   # deep red (disease)
C_CONTROL         = '#1565C0'   # deep blue (control)

# General palette (natbarplot defaults)
C_RED             = '#E8655A'
C_GREEN           = '#6AB56E'
C_BLUE            = '#7EAED3'
C_PURPLE          = '#9B7DB8'
C_ORANGE          = '#E8A952'
C_TEAL            = '#0E8C6A'

# Validation / significance
C_SIGNIFICANT     = '#C62828'
C_NONSIG          = '#BDBDBD'

# ═══════════════════════════════════════════════════════════════
# UNIFIED FONT-SIZE SCALE (use these instead of ad-hoc numbers)
# ═══════════════════════════════════════════════════════════════
F_VALUE   = 28   # data-value labels printed on bars/points
F_LEGEND  = 26   # legend entries
F_ANNOT   = 24   # secondary annotations (n=, axis sub-text)

# Drug mining
C_APPROVED        = '#2ECC71'
C_CLINICAL        = '#F1C40F'
C_PRECLINICAL     = '#E67E22'

# Markers cycling (natbarplot)
MARKERS = ['o', 's', '^', 'D', 'd', 'v', 'p', 'h']

# ═══════════════════════════════════════════════════════════════
# FIGURE SIZE CONSTANTS (svgpic standard)
# ═══════════════════════════════════════════════════════════════

SZ_BAR    = (10, 10)
SZ_WIDE   = (20, 10)
SZ_SQ     = (10, 10)
SZ_HEAT   = (20, 10)
SZ_HBAR   = (10, 10)
SZ_PIE    = (10, 10)
SZ_FOREST = (10, 10)

# ═══════════════════════════════════════════════════════════════
# STYLE APPLICATION (svgpic standard)
# ═══════════════════════════════════════════════════════════════

def apply_style():
    """Apply svgpic publication style — large bold fonts, thick spines."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.weight': 'bold',
        'font.size': 35,
        'axes.labelsize': 37,
        'axes.labelweight': 'bold',
        'axes.titlesize': 40,
        'axes.titleweight': 'bold',
        'axes.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
        'legend.fontsize': 26,
        'legend.frameon': False,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 600,
        'savefig.facecolor': 'white',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'axes.grid': False,
        'mathtext.default': 'regular',
    })

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save(fig, outdir, name):
    """Export figure as SVG + PDF + TIFF + PNG at 600 DPI.

    NO bbox_inches='tight' — output size = figsize × DPI exactly,
    so all panels are integer multiples of 600px (1 inch at 600 DPI).
    Callers should use fig.subplots_adjust() to prevent label clipping.
    """
    for fmt in ('svg', 'pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'{name}.{fmt}'),
                    pad_inches=0, format=fmt)
    fig.savefig(os.path.join(outdir, f'{name}.tiff'),
                pad_inches=0, format='tiff', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f"  Saved: {name}.svg/pdf/tiff/png")


def clean(ax):
    """Remove top/right spines. No grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_legend(ax):
    """Place legend in the right-side whitespace area."""
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)


def adjust_for_legend(fig):
    """Shrink plotting area to leave space for a right-side legend."""
    fig.subplots_adjust(left=0.14, right=0.75, top=0.95, bottom=0.16)


def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def bracket(ax, x1, x2, y, h, p_val, fontsize=32):
    """Draw statistical comparison bracket with significance stars."""
    stars = p_to_stars(p_val)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.7, c='black', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars,
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def hollow_bar(ax, x, height, color, width=0.55, label=None, **kwargs):
    """Draw hollow bar (natbarplot style) — no fill, colored edge."""
    return ax.bar(x, height, width=width, facecolor='none',
                  edgecolor=color, linewidth=2.5, label=label, **kwargs)


def hollow_scatter(ax, x, y, color, marker='o', s=55, jitter=0.12, seed=42):
    """Draw hollow scatter with jitter (natbarplot style)."""
    rng = np.random.default_rng(seed)
    x_jit = np.asarray(x, dtype=float) + rng.uniform(-jitter, jitter, len(x))
    return ax.scatter(x_jit, y, facecolors='none', edgecolors=color,
                      marker=marker, s=s, linewidths=2.0, zorder=3,
                      clip_on=False)


def add_errorbar(ax, x, mean, sem, color='black'):
    """Add SEM error bar (natbarplot style)."""
    ax.errorbar(x, mean, yerr=sem, fmt='none', ecolor=color,
                capsize=6, capthick=2.5, elinewidth=2.5)


def copy_data(src, outdir):
    """Copy raw data file and return SHA-256 hash."""
    dst = os.path.join(outdir, os.path.basename(src))
    shutil.copy2(src, dst)
    return file_hash(dst)


def file_hash(path):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
