#!/usr/bin/env python3
"""
M6 Benchmark: GenKI virtual-KO prediction vs. empirical Trem2-KO differential expression.

Replicates, on the Trem2-KO microglia dataset, the validation reported in
Chen et al. (Nucleic Acids Research 51:6342-6358, 2023): predicted KO-responsive
genes (from GenKI run on WT data) should significantly overlap the empirical DE
genes (from real Trem2-KO vs WT cells), assessed by hypergeometric enrichment
and gene-set AUROC / average precision.

The Trem2-KO microglia dataset is distributed by the GenKI authors via Google Drive
(see tools/GenKI-master/data/README.md):
    WT : microglial_seurat_WT.h5ad   (public, Google Drive file ID 1tG9bUGCsWqhg0hJ94lDLtLl8WLl0hDks)
    KO : microglial_seurat_KO.h5ad   ( Lung / intestine datasets are available from the authors on request. )

EXPECTED DATA LAYOUT (drop files here before running):
    D:/AD-Multi-Omics-Integration/data/genki_benchmark/
        microglial_seurat_WT.h5ad
        microglial_seurat_KO.h5ad       # optional but recommended for the full benchmark
        trem2_DE_genes.csv              # optional pre-computed DE list (gene,col)

USAGE:
    python benchmark_genki_trem2.py
    python benchmark_genki_trem2.py --target-gene Trem2 --epochs 200 --n-permutations 1000
    python benchmark_genki_trem2.py --output-dir output/genki_trem2_benchmark

OUTPUTS (output/genki_trem2_benchmark/):
    predicted_KO_rank.csv        # GenKI KL-divergence distance + rank per gene
    empirical_DE.csv             # WT-vs-KO DE genes (ranked, if KO file present)
    benchmark_metrics.json       # hypergeometric p, AUROC, AUPRC, overlap counts
    benchmark_summary.txt        # human-readable summary
    fig_trem2_benchmark.pdf      # rank-rank / ROC / PR / volcano panels
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# Path setup (robust: derive from this file, no hard-coded /mnt/d paths)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]         # scripts/step4_virtual_knockout/.. == repo root
sys.path.insert(0, str(PROJECT_ROOT))                      # for tools.config_loader
GENKI_DIR = PROJECT_ROOT / "tools" / "GenKI-master"
sys.path.insert(0, str(GENKI_DIR))

DATA_DIR = PROJECT_ROOT / "data" / "genki_benchmark"


def rstr(p):
    return str(p).replace("\\", "/")


# ----------------------------------------------------------------------------
# Step 1 — Run GenKI on WT data, simulate Trem2 KO, get predicted gene ranking
# ----------------------------------------------------------------------------
def run_genki_prediction(wt_h5ad, target_gene, epochs, lr, beta, seed,
                         n_perms, grn_dir, verbose=True):
    """Train VGAE on WT, perturb target_gene, return KL-distance per gene + null."""
    import scanpy as sc
    import GenKI as gk
    from GenKI.preprocesing import build_adata
    from GenKI.dataLoader import DataLoader
    from GenKI.train import VGAE_trainer
    from GenKI import utils

    if verbose:
        print(f"[GenKI] loading WT data from {rstr(wt_h5ad)}")
    adata = build_adata(rstr(wt_h5ad))
    # build_adata uppercases var_names; match target_gene accordingly
    tg = target_gene.upper()
    if tg not in adata.var_names:
        # try a case-insensitive match
        match = [g for g in adata.var_names if g.upper() == tg]
        if not match:
            raise ValueError(f"target gene {target_gene} not in adata.var_names "
                             f"(checked uppercased form {tg})")
        tg = match[0]

    grn_dir.mkdir(parents=True, exist_ok=True)
    wrapper = DataLoader(
        adata,
        target_gene=[tg],
        target_cell=None,
        obs_label="ident",
        GRN_file_dir=rstr(grn_dir),
        rebuild_GRN=True,
        pcNet_name="pcNet_trem2",
        verbose=verbose,
    )
    data_wt = wrapper.load_data()
    data_ko = wrapper.load_kodata()   # simulated KO (target gene zeroed + edges removed)

    if verbose:
        print("[GenKI] training VGAE ...")
    sensei = VGAE_trainer(data_wt, epochs=epochs, lr=lr, beta=beta,
                          seed=seed, log_dir=None, verbose=False)
    sensei.train()

    z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)
    z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
    dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")

    if n_perms and n_perms > 0:
        if verbose:
            print(f"[GenKI] permutation null ({n_perms}) ...")
        null = sensei.pmt(data_ko, n=n_perms, by="KL")
        res = utils.get_generank(data_wt, dis, null)
    else:
        res = utils.get_generank(data_wt, dis, rank=True)

    res = res.copy()
    res.index.name = "gene"
    return res.reset_index()


# ----------------------------------------------------------------------------
# Step 2 — Empirical DE genes from real WT vs KO cells (if KO h5ad available)
# ----------------------------------------------------------------------------
def compute_empirical_DE(wt_h5ad, ko_h5ad, lfc_thresh=0.5, p_thresh=0.05, verbose=True):
    """Wilcoxon rank-sum WT vs KO, return ranked DE gene table."""
    import scanpy as sc
    from GenKI.preprocesing import build_adata

    if verbose:
        print(f"[DE] WT={rstr(wt_h5ad)}  KO={rstr(ko_h5ad)}")
    adata_wt = build_adata(rstr(wt_h5ad))
    adata_ko = build_adata(rstr(ko_h5ad))

    common = adata_wt.var_names.intersection(adata_ko.var_names)
    adata_wt = adata_wt[:, common].copy()
    adata_ko = adata_ko[:, common].copy()
    adata_wt.obs["cond"] = "WT"
    adata_ko.obs["cond"] = "KO"
    merged = sc.concat([adata_wt, adata_ko], axis=0, join="inner")
    sc.pp.log1p(merged, base=2) if "log1p" not in merged.uns else None

    # Wilcoxon (KO vs WT), rank_genes_groups handles ties / sparse
    sc.tl.rank_genes_groups(merged, groupby="cond", groups=["KO"],
                            reference="WT", method="wilcoxon", corr_method="benjamini-hochberg")
    rg = merged.uns["rank_genes_groups"]
    de = pd.DataFrame({k: rg[k]["KO"] for k in rg.dtype.names})
    de = de.rename(columns={"names": "gene", "logfoldchanges": "logFC",
                            "pvals_adj": "pvals_adj", "scores": "score"})
    de["gene"] = de["gene"].astype(str)
    de = de.sort_values("pvals_adj").reset_index(drop=True)
    de["DE"] = (de["pvals_adj"] < p_thresh) & (de["logFC"].abs() > lfc_thresh)
    return de[["gene", "logFC", "pvals_adj", "score", "DE"]]


# ----------------------------------------------------------------------------
# Step 3 — Compare prediction vs empirical truth
# ----------------------------------------------------------------------------
def evaluate(pred_df, de_df, top_k_list=(50, 100, 200), verbose=True):
    """
    pred_df : GenKI output, columns ['gene','dis','rank', possibly 'hit']
              Higher dis == stronger predicted KO response.
    de_df   : columns ['gene','logFC','pvals_adj','DE']  (DE == True/False)
    """
    from scipy.stats import hypergeom
    from sklearn.metrics import roc_auc_score, average_precision_score

    pred = pred_df[["gene"]].copy()
    pred["dis"] = pred_df["dis"].astype(float).values
    # higher distance => stronger predicted response; ensure desc sort
    pred = pred.sort_values("dis", ascending=False).reset_index(drop=True)

    truth = de_df.set_index("gene")["DE"]
    # restrict to genes present in BOTH prediction and truth
    shared = pred["gene"].isin(truth.index)
    pred_s = pred[shared].copy()
    y = truth.loc[pred_s["gene"]].astype(int).values
    scores = pred_s["dis"].values

    N = len(truth)                       # genes tested for DE
    K = int(truth.sum())                 # true DE genes
    n = len(pred_s)                      # shared universe (predicted & measurable)
    if K == 0:
        raise RuntimeError("No empirical DE genes found at the given thresholds; "
                           "loosen --lfc-thresh / --p-thresh or provide a DE list.")

    auroc = roc_auc_score(y, scores)
    auprc = average_precision_score(y, scores)

    out = {"universe_shared": n, "n_true_DE": K,
           "AUROC": float(auroc), "AUPRC": float(auprc),
           "topK": {}}
    for k in top_k_list:
        kk = min(k, len(pred_s))
        top_pred = set(pred_s["gene"].iloc[:kk])
        k_hits = int(sum(truth.loc[list(top_pred)]))
        # hypergeometric P(>= k_hits | N n K)
        p_hyp = float(hypergeom.sf(k_hits - 1, N, K, kk)) if kk else float("nan")
        out["topK"][kk] = {"predicted_top": kk, "DE_overlap": k_hits,
                           "hypergeom_p": p_hyp,
                           "fold_enrichment": float((k_hits / max(kk, 1)) / (K / max(N, 1)))}
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=rstr(DATA_DIR))
    ap.add_argument("--target-gene", default="Trem2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=8096)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--lfc-thresh", type=float, default=0.5)
    ap.add_argument("--p-thresh", type=float, default=0.05)
    ap.add_argument("--output-dir", default=rstr(PROJECT_ROOT / "output" / "genki_trem2_benchmark"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grn_dir = out_dir / "GRNs"

    wt = data_dir / "microglial_seurat_WT.h5ad"
    ko = data_dir / "microglial_seurat_KO.h5ad"
    de_csv = data_dir / "trem2_DE_genes.csv"

    if not wt.exists():
        sys.exit(f"[ERROR] WT data not found at {rstr(wt)}.\n"
                 f"        Download from Google Drive file ID 1tG9bUGCsWqhg0hJ94lDLtLl8WLl0hDks\n"
                 f"        (see tools/GenKI-master/data/README.md) into data/genki_benchmark/.")

    # ---- Step 1: GenKI prediction on WT, simulated Trem2 KO ----
    pred = run_genki_prediction(
        wt, args.target_gene, args.epochs, args.lr, args.beta,
        args.seed, args.n_permutations, grn_dir)
    pred.to_csv(out_dir / "predicted_KO_rank.csv", index=False)
    print(f"[OK] predicted_KO_rank.csv  ({len(pred)} genes)")

    # ---- Step 2: empirical truth ----
    if de_csv.exists():
        print(f"[DE] loading pre-computed DE list: {rstr(de_csv)}")
        de = pd.read_csv(de_csv)
        # accept 'gene' + 'logFC'/'p' columns flexibly
        if "DE" not in de.columns:
            pcol = "pvals_adj" if "pvals_adj" in de.columns else ("p" if "p" in de.columns else "pvals_adj")
            lcol = "logFC" if "logFC" in de.columns else ("log2FoldChange" if "log2FoldChange" in de.columns else "logFC")
            de["DE"] = (de[pcol] < args.p_thresh) & (de[lcol].abs() > args.lfc_thresh)
    elif ko.exists():
        de = compute_empirical_DE(wt, ko, args.lfc_thresh, args.p_thresh)
    else:
        sys.exit(f"[ERROR] No empirical truth available.\n"
                 f"        Provide either microglial_seurat_KO.h5ad (to compute DE) or\n"
                 f"        trem2_DE_genes.csv (pre-computed DE list) in {rstr(data_dir)}.\n"
                 f"        Lung/intestine KO datasets are available from GenKI authors on request.")
    de.to_csv(out_dir / "empirical_DE.csv", index=False)
    print(f"[OK] empirical_DE.csv  ({int(de['DE'].sum())} DE genes of {len(de)})")

    # ---- Step 3: evaluate ----
    metrics = evaluate(pred, de)
    metrics["target_gene"] = args.target_gene
    metrics["parameters"] = vars(args)
    with open(out_dir / "benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # human-readable summary
    lines = ["GenKI Trem2-KO benchmark summary", "=" * 40]
    lines.append(f"target gene      : {args.target_gene}")
    lines.append(f"shared universe  : {metrics['universe_shared']}")
    lines.append(f"true DE genes    : {metrics['n_true_DE']}")
    lines.append(f"AUROC            : {metrics['AUROC']:.4f}")
    lines.append(f"AUPRC            : {metrics['AUPRC']:.4f}")
    lines.append("")
    lines.append("Top-K overlap (hypergeometric enrichment):")
    for k, v in metrics["topK"].items():
        lines.append(f"  top {k:>4}: overlap={v['DE_overlap']:>3}  "
                     f"fold={v['fold_enrichment']:.2f}  "
                     f"p={v['hypergeom_p']:.2e}")
    summary = "\n".join(lines) + "\n"
    (out_dir / "benchmark_summary.txt").write_text(summary)
    print(summary)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve

        truth = de.set_index("gene")["DE"]
        pred_sorted = pred.sort_values("dis", ascending=False)
        shared = pred_sorted["gene"].isin(truth.index)
        ps = pred_sorted[shared]
        y = truth.loc[ps["gene"]].astype(int).values
        s = ps["dis"].values
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].scatter(range(1, len(ps) + 1), ps["dis"].values, s=6,
                        c=y, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_xlabel("predicted rank"); axes[0].set_ylabel("KL distance (KO vs WT)")
        axes[0].set_title("Predicted KO response"); axes[0].set_xscale("log")
        axes[1].plot(fpr, tpr, label=f"AUROC={metrics['AUROC']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.5)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC")
        axes[1].legend()
        axes[2].plot(rec, prec, label=f"AUPRC={metrics['AUPRC']:.3f}")
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("Precision-Recall")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_trem2_benchmark.pdf")
        print(f"[OK] fig_trem2_benchmark.pdf")
    except Exception as e:
        print(f"[WARN] figure not generated: {e}")


if __name__ == "__main__":
    main()
