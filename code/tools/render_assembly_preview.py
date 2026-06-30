#!/usr/bin/env python3
"""Render the SAME grid layout as assemble_figures_pptx.py to PNG, so the
assembly can be visually verified without PowerPoint. Pure PIL — mirrors the
unit math of the pptx builder so what you see == the .pptx layout."""
import os
from PIL import Image, ImageDraw, ImageFont

PANEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', '_panels')
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures', '_assembly_qa')

PX_PER_UNIT = 700   # px per height-unit
LABEL_PX = 60

RATIO = {f'fig{i}{c}': r for i, cs in [
    (1, {'a': 1, 'b': 3, 'c': 1, 'd': 3, 'e': 2, 'f': 2}),
    (2, {'a': 2, 'b': 2, 'c': 2, 'd': 2}),
    (3, {'a': 1, 'b': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 2}),
    (4, {'a': 2, 'b': 2, 'c': 2, 'd': 2}),
    (5, {'a': 2, 'b': 2, 'c': 2, 'd': 2}),
    (6, {'a': 1, 'b': 2, 'c': 3, 'd': 1}),
    (7, {'a': 2, 'b': 2, 'c': 2, 'd': 2}),
] for c, r in cs.items()}

ROWS = {
    1: [['fig1a', 'fig1b'], ['fig1c', 'fig1d'], ['fig1e', 'fig1f']],
    2: [['fig2a', 'fig2b'], ['fig2c', 'fig2d']],
    3: [['fig3a', 'fig3b', 'fig3c'], ['fig3d', 'fig3e', 'fig3f']],
    4: [['fig4a', 'fig4b'], ['fig4c', 'fig4d']],
    5: [['fig5a', 'fig5b'], ['fig5c', 'fig5d']],
    7: [['fig7a', 'fig7b'], ['fig7c', 'fig7d']],
}


def load(panel):
    return Image.open(os.path.join(PANEL_DIR, f'{panel}.png')).convert('RGB')


def fit(im, cw, ch):
    ar = im.width / im.height
    if cw / ch > ar:
        h, w = ch, int(round(ch * ar))
    else:
        w, h = cw, int(round(cw / ar))
    return im.resize((w, h), Image.LANCZOS)


def paste_with_label(canvas, panel, x, y, cw, ch, label, font):
    im = fit(load(panel), cw, ch)
    canvas.paste(im, (x, y))
    d = ImageDraw.Draw(canvas)
    d.text((x + 14, y + 6), label, fill='black', font=font)


GAP = 30  # px gap between panels


def build_rows(canvas, rows, U, x0, y0, font):
    for ri, row in enumerate(rows):
        row_top = y0 + ri * U + ri * GAP
        x = x0
        for pi, panel in enumerate(row):
            if pi > 0:
                x += GAP
            cw = RATIO[panel] * U - GAP
            ch = U - GAP if ri < len(rows) - 1 else U
            paste_with_label(canvas, panel, x, row_top, cw, ch, panel[-1], font)
            x += RATIO[panel] * U


def build_fig6(canvas, U, x0, y0, font):
    half = U
    paste_with_label(canvas, 'fig6a', x0, y0, 1 * U, half, 'a', font)
    paste_with_label(canvas, 'fig6b', x0 + 1 * U, y0, 2 * U, half, 'b', font)
    paste_with_label(canvas, 'fig6c', x0, y0 + half, 3 * U, half, 'c', font)
    paste_with_label(canvas, 'fig6d', x0 + 3 * U, y0, 2 * U, 2 * U, 'd', font)


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        font = ImageFont.truetype("arialbd.ttf", LABEL_PX)
    except Exception:
        font = ImageFont.load_default()
    for fig_num in [1, 2, 3, 4, 5, 6, 7]:
        if fig_num == 6:
            gw, gh = 5 * PX_PER_UNIT, 2 * PX_PER_UNIT
        else:
            gw = 4 * PX_PER_UNIT
            gh = len(ROWS[fig_num]) * PX_PER_UNIT
        pad = LABEL_PX + 20
        canvas = Image.new('RGB', (gw + 2 * pad, gh + 2 * pad), 'white')
        x0, y0 = pad, pad
        if fig_num == 6:
            build_fig6(canvas, PX_PER_UNIT, x0, y0, font)
        else:
            build_rows(canvas, ROWS[fig_num], PX_PER_UNIT, x0, y0, font)
        path = os.path.join(OUT, f'figure{fig_num}_assembly.png')
        # downscale a touch for the vision check
        canvas.thumbnail((1600, 1600))
        canvas.save(path)
        print(f'figure{fig_num}: {canvas.size} -> {path}')


if __name__ == '__main__':
    main()
