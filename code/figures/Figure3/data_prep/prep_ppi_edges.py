# prep_ppi_edges.py — extract top-N PPI propagation edges per KO for Figure 3 panels d/e.
# For each KO's gene_ranking.csv, keep the top-N targets by Effect_Score,
# annotate which targets are shared across KOs (for node sizing in the network).
import os
import glob
import pandas as pd

ROOT = ros.path.join(str(PROJECT_ROOT), "output/step4_virtual_knockout/PPI_propagation")
OUT_DIR = ros.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/data")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N = 5

def build(direction):
    pattern = os.path.join(ROOT, direction, "proteomics_*_gene_ranking.csv")
    frames = []
    for f in glob.glob(pattern):
        ko = os.path.basename(f).replace("proteomics_", "").replace("_gene_ranking.csv", "")
        df = pd.read_csv(f)
        df = df.sort_values("Effect_Score", ascending=False).head(TOP_N).copy()
        df["KO_gene"] = ko
        df["direction"] = direction.capitalize()
        frames.append(df[["KO_gene", "Gene", "Effect_Score", "Z_score", "direction"]])
    edges = pd.concat(frames, ignore_index=True)
    edges = edges.rename(columns={"Gene": "target", "Effect_Score": "effect_score"})
    # shared = target appears for >1 KO in this direction
    counts = edges.groupby("target")["KO_gene"].nunique()
    edges["target_shared_n"] = edges["target"].map(counts).fillna(1).astype(int)
    edges["shared"] = edges["target_shared_n"] > 1
    return edges

fwd = build("forward")
rev = build("reverse")

fwd_out = os.path.join(OUT_DIR, "fig3b_ppi_fwd_edges.csv")
rev_out = os.path.join(OUT_DIR, "fig3c_ppi_rev_edges.csv")
fwd.to_csv(fwd_out, index=False)
rev.to_csv(rev_out, index=False)
print(f"Saved {fwd_out}  ({len(fwd)} edges, {fwd['target'].nunique()} unique targets, {fwd['shared'].sum()} shared-edge rows)")
print(f"Saved {rev_out}  ({len(rev)} edges, {rev['target'].nunique()} unique targets, {rev['shared'].sum()} shared-edge rows)")
print("Forward shared hubs:", fwd[fwd['shared']]['target'].unique().tolist())
print("Reverse shared hubs:", rev[rev['shared']]['target'].unique().tolist())
