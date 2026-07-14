# run_all.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from panel_b_umap import main as b
from panel_c_training import main as c
from panel_d_fair_mapping import main as d
from panel_e_jacobian_auc import main as e
from panel_f_trajectory import main as f
from panel_g_reversibility import main as g
from panel_h_multimetric import main as h

if __name__ == "__main__":
    print("=== Figure 1 ===")
    b(); c(); d(); e(); f(); g(); h()
    print("=== Done ===")
