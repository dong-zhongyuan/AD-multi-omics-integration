#!/usr/bin/env python3
"""Render the pptx grid to PNG for QA. Each row's panels share an equal height;
row height chosen so the widest row spans the target width; rows stacked with
a gap. Mirrors assemble_r_figures_pptx.py math."""
import os
from PIL import Image, ImageDraw

SRC = os.path.join(os.path.dirname(__file__), '..', 'figures', 'r_figures')
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures', '_r_qa2')
TARGET_W = 2400        # px width of the rendered figure
GAP_FRAC = 0.04        # gap as fraction of row height
LABEL_PX = 36

ROWS = {
    1: [['fig1a', 'fig1b'], ['fig1c', 'fig1d'], ['fig1e', 'fig1f']],
    2: [['fig2a', 'fig2b'], ['fig2c', 'fig2d']],
    3: [['fig3a', 'fig3b', 'fig3c'], ['fig3d', 'fig3e', 'fig3f']],
    4: [['fig4a', 'fig4b'], ['fig4c', 'fig4d']],
    5: [['fig5a', 'fig5b'], ['fig5c', 'fig5d']],
    6: [['fig6a', 'fig6b'], ['fig6c', 'fig6d']],
    7: [['fig7a', 'fig7b'], ['fig7c', 'fig7d']],
}


def img(panel):
    fig = 'fig' + panel[3]
    im = Image.open(os.path.join(SRC, fig, panel + '.png'))
    return im.width / im.height, im


def render(rows):
    ars = {p: img(p)[0] for row in rows for p in row}
    # row width (at height=1) = sum of aspect ratios + gaps
    gap = GAP_FRAC
    row_w_at_h1 = [sum(ars[p] for p in row) + gap * (len(row) - 1) for row in rows]
    max_row_w = max(row_w_at_h1)
    h = TARGET_W / max_row_w          # unit height in px
    gap_px = h * gap
    total_h = len(rows) * h + (len(rows) - 1) * gap_px
    canvas = Image.new('RGB', (TARGET_W, int(total_h)), 'white')
    d = ImageDraw.Draw(canvas)
    y = 0
    for ri, row in enumerate(rows):
        rw = sum(ars[p] for p in row) + gap * (len(row) - 1)
        x = (TARGET_W - rw * h) / 2
        for pi, panel in enumerate(row):
            if pi > 0:
                x += gap_px
            _, im = img(panel)
            w = h * ars[panel]
            canvas.paste(im.resize((int(w), int(h)), Image.LANCZOS),
                         (int(x), int(y)))
            d.text((int(x) + 10, int(y) + 6), panel[-1], fill='black')
            x += w
        y += h + gap_px
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    for fig_num in [1, 2, 3, 4, 5, 6, 7]:
        canvas = render(ROWS[fig_num])
        canvas.thumbnail((1600, 1600))
        path = os.path.join(OUT, f'figure{fig_num}.png')
        canvas.save(path)
        print(f'figure{fig_num}: {canvas.size}')


if __name__ == '__main__':
    main()
