#!/usr/bin/env python3
"""Regenerate all panels for figure2 from *_raw to SVG/PDF/TIFF."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
PANELS = ['fig2a', 'fig2b', 'fig2c', 'fig2d']
def run_script(path):
    subprocess.run([sys.executable, str(path)], cwd=BASE, check=True)
def main():
    for stem in PANELS:
        run_script(BASE / f'{stem}_process.py')
        run_script(BASE / f'{stem}_plot.py')
    print(f'Regenerated {len(PANELS)} panels for {BASE.name}')
if __name__ == '__main__':
    main()
