# run_all.py — regenerate every Figure 4 panel (a-j), 10 panels.
import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PANELS = [
    'panel_a_schematic',
    'panel_b_adni_cohort',
    'panel_c_top_auc',
    'panel_d_panel_auc',
    'panel_e_roc',
    'panel_f_cox_proteomics',
    'panel_g_l1_coefficients',
    'panel_h_mapt_km',
    'panel_i_cox_transcriptomics',
    'panel_j_expression_3d',
]

if __name__ == '__main__':
    print('=== Figure 4 (10 panels a-j) ===')
    for name in PANELS:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        print(f'  -> {name} done')
    print('=== Done ===')
