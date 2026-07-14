# run_all.py — regenerate every Figure 5 panel (a-h), 8 panels.
import os, sys, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PANELS = [
    'panel_a_schematic',
    'panel_b_transcriptomics_forest',
    'panel_c_proteomics_forest',
    'panel_d_druggable_targets',
    'panel_e_km_mmp9_cxcr2',
    'panel_f_druggability_tiers',
    'panel_g_pathway_complementarity',
    'panel_h_drug_bubble',
]
if __name__ == '__main__':
    print('=== Figure 5 (8 panels a-h) ===')
    for name in PANELS:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        print(f'  -> {name} done')
    print('=== Done ===')
