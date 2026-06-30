#!/usr/bin/env python3
"""Regenerate all panels for figure5 from *_raw to SVG/PDF/TIFF."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
PANELS = ['fig5a', 'fig5b', 'fig5c', 'fig5d']
# fig5c uses the ROC-curve processor (cross_val_predict -> fpr/tpr points),
# not the 4-row AUC summary processor
PROCESSOR = {'fig5c': 'fig5c_roc_process'}
def run_script(path):
    subprocess.run([sys.executable, str(path)], cwd=BASE, check=True)
def main():
    for stem in PANELS:
        run_script(BASE / f'{PROCESSOR.get(stem, stem + "_process")}.py')
        run_script(BASE / f'{stem}_plot.py')
    print(f'Regenerated {len(PANELS)} panels for {BASE.name}')
if __name__ == '__main__':
    main()
