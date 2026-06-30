#!/usr/bin/env python3
"""Regenerate all panels for figure6 from *_raw to SVG/PDF/TIFF."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PANELS = ['fig6a', 'fig6b', 'fig6c', 'fig6d']

def run_script(path: Path) -> None:
    subprocess.run([sys.executable, str(path)], cwd=BASE, check=True)

def main() -> None:
    for stem in PANELS:
        run_script(BASE / f'{stem}_process.py')
        run_script(BASE / f'{stem}_plot.py')
    print(f'Regenerated {len(PANELS)} panels for {BASE.name}')

if __name__ == '__main__':
    main()
