# run_all.py — regenerate every Figure 3 panel (a-j), 10 panels mixing Python and R.
import os
import sys
import importlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RSCRIPT = r"C:/Program Files/R/R-4.4.3/bin/Rscript.exe"

PANELS = [
    ('panel_a_schematic',        'py'),
    ('panel_b_ppi_circular_fwd', 'R'),
    ('panel_c_ppi_circular_rev', 'R'),
    ('panel_d_radial_genki',     'R'),
    ('panel_e_confidence_scatter','py'),
    ('panel_f_scenic_grid',      'py'),
    ('panel_g_concordance',      'py'),
    ('panel_h_ridge_kl',         'R'),
    ('panel_i_sankey_bubble',    'R'),
    ('panel_j_chord_ko',         'R'),
]

def run_R(name):
    script = os.path.join(HERE, name + '.R')
    result = subprocess.run([RSCRIPT, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  !! {name} FAILED (R exit {result.returncode})')
        print(result.stderr[-1500:])
    else:
        out = result.stdout.strip().splitlines()
        print(out[-1] if out else f'  -> {name} done')

def run_py(name):
    mod = importlib.import_module(name)
    importlib.reload(mod)

if __name__ == '__main__':
    print('=== Figure 3 (10 panels a-j, Python + R) ===')
    for name, kind in PANELS:
        try:
            if kind == 'R': run_R(name)
            else:
                run_py(name); print(f'  -> {name} done')
        except Exception as e:
            print(f'  !! {name} FAILED: {e}')
    print('=== Done ===')
