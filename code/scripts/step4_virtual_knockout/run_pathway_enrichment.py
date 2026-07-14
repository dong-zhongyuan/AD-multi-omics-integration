#!/usr/bin/env python3
"""
Pathway enrichment of virtual-knockout perturbed target sets.

For each VK method (PPI_propagation, GenKI_NO3) and each direction (forward/reverse),
define the "perturbed target set" as the union over all KOs of targets whose
perturbation score clears a Z-score threshold, then run over-representation
analysis (ORA) via gseapy/Enrichr against GO Biological Process.

Outputs (per method x direction):
  enrichment_<method>_<direction>.csv
    columns: Gene_set, Term, Overlap, P-value, Adjusted P-value, Odds Ratio,
             Combined Score, Genes   (sorted by Adjusted P-value; top N kept)

Usage:
  python run_pathway_enrichment.py
  python run_pathway_enrichment.py --z-threshold 2.5 --top-n 15
  python run_pathway_enrichment.py --methods PPI_propagation
"""
import os
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import gseapy as gp

# ---------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VK_DIR  = os.path.join(PROJECT, "output", "step4_virtual_knockout")
OUT_DIR = os.path.join(VK_DIR, "pathway_enrichment")

GENE_SETS = "GO_Biological_Process_2023"
ORGANISM  = "human"
DEFAULT_Z = 2.0
DEFAULT_TOPN = 12

# Each method declares how to resolve the directory that holds its gene_ranking
# files for a given direction. PPI uses <method>/<direction>/; GenKI uses
# <method>/ for forward and <method>_reverse/ for reverse.
def ppi_dir(direction):
    return os.path.join(VK_DIR, "PPI_propagation", direction)

def genki_dir(direction):
    sub = "GenKI_NO3" if direction == "forward" else "GenKI_NO3_reverse"
    return os.path.join(VK_DIR, sub)

# (dir_resolver, ranking_filename_pattern, label)
METHODS = [
    (ppi_dir,   "proteomics_*_gene_ranking.csv",      "PPI"),
    (genki_dir, "proteomics_*_gene_ranking.csv",      "GenKI_prot"),
    (genki_dir, "transcriptomics_*_gene_ranking.csv", "GenKI_trans"),
]
DIRECTIONS = [("forward", "fwd"), ("reverse", "rev")]


def collect_perturbed(dir_resolver, pattern, direction, z_thr):
    """Union of genes with Z_score > z_thr across all KOs of this method/direction."""
    base = dir_resolver(direction)
    if not os.path.isdir(base):
        return [], 0
    files = glob.glob(os.path.join(base, pattern))
    genes = set()
    for f in files:
        try:
            df = pd.read_csv(f)
            if "Z_score" in df.columns:
                genes |= set(df.loc[df["Z_score"] > z_thr,
                                    "Gene"].astype(str).tolist())
        except Exception:
            continue
    return sorted(genes), len(files)


def run_enrichment(genes, label, out_dir, top_n):
    if not genes:
        print(f"  [{label}] no perturbed genes, skipping")
        return None
    try:
        enr = gp.enrichr(gene_list=genes, gene_sets=GENE_SETS,
                         organism=ORGANISM, outdir=None, cutoff=1.0)
    except Exception as e:
        print(f"  [{label}] Enrichr failed: {e}")
        return None
    res = enr.results.sort_values("Adjusted P-value").head(top_n).reset_index(drop=True)
    out = os.path.join(out_dir, f"enrichment_{label}.csv")
    res.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-threshold", type=float, default=DEFAULT_Z,
                    help=f"Z-score cutoff for 'perturbed target' (default {DEFAULT_Z})")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOPN,
                    help=f"Top-N terms kept per result (default {DEFAULT_TOPN})")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Subset of method labels to run (default: all)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Pathway enrichment (ORA, {GENE_SETS})")
    print(f"  Z-threshold: {args.z_threshold}   top-N: {args.top_n}")
    print(f"  output: {OUT_DIR}\n")

    summary = []
    for dir_resolver, pattern, label in METHODS:
        if args.methods and label not in args.methods:
            continue
        for direction, short in DIRECTIONS:
            full_label = f"{label}_{short}"
            genes, n_ko = collect_perturbed(dir_resolver, pattern, direction,
                                            args.z_threshold)
            print(f"[{full_label}] {n_ko} KOs -> {len(genes)} perturbed genes")
            if not genes:
                continue
            out = run_enrichment(genes, full_label, OUT_DIR, args.top_n)
            if out:
                top_term = pd.read_csv(out).iloc[0]
                print(f"  saved {os.path.basename(out)}; top term: "
                      f"{top_term['Term'][:50]}... (adj p={top_term['Adjusted P-value']:.2e})")
                summary.append({"method": label, "direction": direction,
                                "n_ko": n_ko, "n_genes": len(genes),
                                "top_term": top_term["Term"],
                                "top_adj_p": top_term["Adjusted P-value"],
                                "output": os.path.basename(out)})
    if summary:
        smry = pd.DataFrame(summary)
        smry.to_csv(os.path.join(OUT_DIR, "_enrichment_summary.csv"), index=False)
        print(f"\nSummary -> {os.path.join(OUT_DIR, '_enrichment_summary.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
