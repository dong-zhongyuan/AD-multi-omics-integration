# prep_sankey_bubble.py — build gene→pathway edgelist for the sankey-bubble panel.
# Output: fig_sankey_edges.csv  (gene, pathway, direction)
#         fig_bubble_summary.csv (pathway, direction, n_genes, odds_ratio, adj_p, hit_ratio)
#
# Reads enrichment results from the MAIN PIPELINE step4 output (canonical source):
#   output/step4_virtual_knockout/pathway_enrichment/enrichment_PPI_{fwd,rev}.csv
# Strategy: for each pathway's enrichment gene list, keep only genes that actually
# appear in the PPI perturbation target set (Z>2) of the corresponding direction.
# This (a) keeps the sankey readable, (b) ties enrichment back to the KO data,
# (c) avoids drawing 200+ gene nodes per pathway.
import os
import glob
import pandas as pd

PROJECT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VK_DIR    = os.path.join(PROJECT, "output", "step4_virtual_knockout")
PPI_DIR   = os.path.join(VK_DIR, "PPI_propagation")
ENR_DIR   = os.path.join(VK_DIR, "pathway_enrichment")    # canonical pipeline output
DATA      = os.path.join(PROJECT, "output", "Figures_final", "Figure3", "data")

PERTURBED = {}
for direction in ("forward", "reverse"):
    genes = set()
    for f in glob.glob(os.path.join(PPI_DIR, direction, "proteomics_*_gene_ranking.csv")):
        df = pd.read_csv(f)
        genes |= set(df.loc[df["Z_score"] > 2, "Gene"].astype(str))
    PERTURBED[direction] = genes
    print(f"{direction}: {len(genes)} perturbed targets (Z>2)")

edges = []
summary = []
for direction, label, short in [("forward", "Forward", "fwd"), ("reverse", "Reverse", "rev")]:
    # Read from canonical pipeline location (enrichment_PPI_{short}.csv)
    enr = pd.read_csv(os.path.join(ENR_DIR, f"enrichment_PPI_{short}.csv")).head(8)
    perturbed = PERTURBED[direction]
    for _, r in enr.iterrows():
        term = r["Term"]
        genes_in_term = [g.strip() for g in str(r["Genes"]).split(";")]
        # keep only perturbed genes that appear in this pathway
        kept = [g for g in genes_in_term if g in perturbed]
        # cap per pathway to avoid huge nodes — keep up to 8
        kept = kept[:8]
        for g in kept:
            edges.append({"gene": g, "pathway": term, "direction": label})
        # parse Overlap "n/d"
        ov = str(r["Overlap"]).split("/")
        hit_ratio = float(ov[0]) / float(ov[1]) if len(ov) == 2 and float(ov[1]) > 0 else 0
        summary.append({
            "pathway": term, "direction": label,
            "n_genes_overlap": int(ov[0]) if ov[0].isdigit() else 0,
            "n_kept_in_sankey": len(kept),
            "odds_ratio": float(r["Odds Ratio"]),
            "adj_p": float(r["Adjusted P-value"]),
            "hit_ratio": hit_ratio,
        })

edges = pd.DataFrame(edges)
summary = pd.DataFrame(summary)
edges.to_csv(os.path.join(DATA, "fig_sankey_edges.csv"), index=False)
summary.to_csv(os.path.join(DATA, "fig_bubble_summary.csv"), index=False)
print(f"edges: {len(edges)} (gene->pathway), {edges['gene'].nunique()} unique genes")
print(f"summary: {len(summary)} pathways")
print("per pathway n_kept_in_sankey:")
print(summary[["pathway","direction","n_kept_in_sankey"]].to_string(index=False))
