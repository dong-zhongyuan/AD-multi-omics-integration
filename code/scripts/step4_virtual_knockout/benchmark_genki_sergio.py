#!/usr/bin/env python3
"""
M6 Benchmark: GenKI virtual-KO on a semi-synthetic network with known ground truth.

This reproduces, in a fully self-contained way (no external download required),
the SERGIO-style validation logic used by Chen et al. (Nucleic Acids Research
51:6342-6358, 2023) to benchmark GenKI against scTenifoldKnk:

  1. Build a gene regulatory network with KNOWN edges (ground-truth GRN).
  2. Sample wild-type single-cell expression counts from a linear-Gaussian
     structural equation model (a standard SERGIO-like generative model).
  3. Run each method (GenKI VGAE, scTenifoldKnk) to score every gene's
     response to a held-out "knockout" gene.
  4. Score each method's ability to recover the held-out gene's TRUE children
     (the ground-truth GRN) via AUROC and average precision (AP).
  5. Also include two baselines (random scores, and Pearson correlation with
     the KO gene) to confirm the discriminative bar.

Because the ground-truth GRN is constructed by us, no external dataset is needed.
The script only depends on packages already in the pipeline environment
(torch, torch_geometric, scanpy, scTenifold, sklearn, scipy).

USAGE:
    python benchmark_genki_sergio.py
    python benchmark_genki_sergio.py --n-genes 200 --n-tfs 20 --n-cells 500 --n-kos 10
    python benchmark_genki_sergio.py --output-dir output/genki_sergio_benchmark

OUTPUTS (output/genki_sergio_benchmark/):
    benchmark_metrics.csv       # per-KO AUROC/AP for each method
    benchmark_summary.txt       # mean +/- SD per method
    fig_sergio_benchmark.pdf    # ROC/PR curves for the first few KOs + boxplots
    grn_truth.csv               # the ground-truth edges
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]         # scripts/step4_virtual_knockout/.. == repo root
sys.path.insert(0, str(PROJECT_ROOT))                       # for tools.config_loader
GENKI_DIR = PROJECT_ROOT / "tools" / "GenKI-master"
sys.path.insert(0, str(GENKI_DIR))

# top-level imports that depend on the above sys.path entries
import scanpy as sc                                       # noqa: E402
import anndata as ad                                      # noqa: E402
import GenKI as gk                                        # noqa: E402
from GenKI.preprocesing import build_adata                # noqa: E402
from GenKI.dataLoader import DataLoader                   # noqa: E402
from GenKI.train import VGAE_trainer                      # noqa: E402
from scTenifold import scTenifoldKnk                      # noqa: E402


def rstr(p):
    return str(p).replace("\\", "/")


# ----------------------------------------------------------------------------
# Step 1 — Build a ground-truth GRN + sample WT expression
# ----------------------------------------------------------------------------
def build_truth_and_data(n_genes, n_tfs, n_children, n_cells, seed=0,
                         coupling=0.6, noise_sd=0.5, verbose=True):
    """
    Linear-Gaussian SEM:  TF -> child edges, x_child = sum_parents(coupling*x) + N(0,noise_sd^2).
    Returns truth (directed adjacency, n_genes x n_genes), WT counts (cells x genes),
    tf_index list, and child-map (tf -> set(child indices)).
    """
    rng = np.random.default_rng(seed)
    tf_idx = list(range(n_tfs))                       # first n_tfs genes are TFs
    truth = np.zeros((n_genes, n_genes), dtype=int)   # truth[parent, child] = 1
    child_map = {t: set() for t in tf_idx}
    for t in tf_idx:
        # choose n_children children from the non-TF pool (or all genes) without self
        pool = [g for g in range(n_genes) if g != t]
        kids = rng.choice(pool, size=min(n_children, len(pool)), replace=False)
        truth[t, kids] = 1
        child_map[t].update(kids.tolist())

    # sample WT expression by ancestral sampling in topological order
    # here TFs are independent; children depend on TFs only (depth-2 DAG) -> easy
    X = np.zeros((n_cells, n_genes), dtype=float)
    X[:, tf_idx] = rng.normal(loc=0.0, scale=1.0, size=(n_cells, n_tfs))   # TF baselines
    for g in range(n_genes):
        parents = np.where(truth[:, g])[0]
        if len(parents) == 0 and g not in tf_idx:
            X[:, g] = rng.normal(0.0, 1.0, size=n_cells)                   # leaf w/o parent
        elif len(parents) > 0:
            X[:, g] = coupling * X[:, parents].sum(axis=1) + rng.normal(0.0, noise_sd, n_cells)

    # non-negative pseudo-counts (SERGIO-like counts)
    Xc = np.clip(X - X.min(), 0, None) + 0.1
    if verbose:
        print(f"[sim] genes={n_genes} TFs={n_tfs} children/TF={n_children} "
              f"cells={n_cells} edges={int(truth.sum())}")
    return truth, Xc, tf_idx, child_map


# ----------------------------------------------------------------------------
# Step 2 — GenKI score for one KO gene
# ----------------------------------------------------------------------------
def genki_ko_score(adata, ko_gene, epochs, lr, beta, seed, grn_dir, verbose=False):
    """Return dict gene-> KL distance (higher = stronger predicted KO response)."""
    grn_dir.mkdir(parents=True, exist_ok=True)
    wrapper = DataLoader(adata, target_gene=[ko_gene], target_cell=None,
                         obs_label="ident", GRN_file_dir=rstr(grn_dir),
                         rebuild_GRN=True, pcNet_name=f"pcNet_sergio", verbose=verbose)
    data = wrapper.load_data()
    data_ko = wrapper.load_kodata()
    sensei = VGAE_trainer(data, epochs=epochs, lr=lr, beta=beta, seed=seed,
                          log_dir=None, verbose=False)
    sensei.train()
    z_mu, z_std = sensei.get_latent_vars(data)
    z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
    dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu, z_std, by="KL")
    genes = list(adata.var_names)
    return dict(zip(genes, np.asarray(dis).ravel()))


# ----------------------------------------------------------------------------
# Step 3 — scTenifoldKnk score for one KO gene
# ----------------------------------------------------------------------------
def sctenifold_ko_score(adata_df, ko_gene, n_neighbors=5, verbose=False):
    """Return dict gene -> |FC| (scTenifoldKnk drift).

    scTenifoldKnk builds multiple WGCNAs which is slow; for the benchmark we only
    need the per-gene drift score, so we keep the default manifold settings but
    expose neighbor count for control.
    """
    knk = scTenifoldKnk(data=adata_df,
                        qc_kws={"min_lib_size": 1, "min_percent": 0.0})
    knk.run_step("qc")
    knk.run_step("nc", n_cpus=1)
    knk.run_step("td")
    knk.run_step("ko", ko_genes=[ko_gene], ko_method="default")
    knk.run_step("ma")
    knk.run_step("dr", sorted_by="adjusted p-value")
    res = knk.d_regulation
    return dict(zip(res["Gene"], res["FC"].abs().to_numpy()))


# ----------------------------------------------------------------------------
# Step 4 — evaluate recovery of true children
# ----------------------------------------------------------------------------
def score_recovery(score_dict, labels, gene_names):
    from sklearn.metrics import roc_auc_score, average_precision_score
    s = np.array([score_dict.get(g, 0.0) for g in gene_names], dtype=float)
    y = np.array(labels, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-genes", type=int, default=100)
    ap.add_argument("--n-tfs", type=int, default=10)
    ap.add_argument("--n-children", type=int, default=5)
    ap.add_argument("--n-cells", type=int, default=300)
    ap.add_argument("--n-kos", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=8096)
    ap.add_argument("--skip-knk", action="store_true",
                    help="skip scTenifoldKnk (slow) — GenKI + baselines only")
    ap.add_argument("--output-dir",
                    default=rstr(PROJECT_ROOT / "output" / "genki_sergio_benchmark"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grn_dir = out_dir / "GRNs"

    # ---- build data ----
    truth, Xc, tf_idx, child_map = build_truth_and_data(
        args.n_genes, args.n_tfs, args.n_children, args.n_cells, seed=args.seed)

    gene_names = [f"g{i}" for i in range(args.n_genes)]

    # build adata for GenKI (uppercased var names, layers['norm'])
    adata = ad.AnnData(Xc)
    adata.var_names = gene_names
    adata.obs_names = [f"c{i}" for i in range(args.n_cells)]
    adata.obs["ident"] = "cell"
    adata.layers["norm"] = Xc
    adata = build_adata(adata, uppercase=False)

    # dataframe for scTenifoldKnk (genes x cells)
    adata_df = pd.DataFrame(Xc.T, index=gene_names, columns=adata.obs_names)

    # save truth
    edges = [(p, c) for p in range(args.n_genes) for c in range(args.n_genes) if truth[p, c]]
    pd.DataFrame(edges, columns=["parent", "child"]).to_csv(
        out_dir / "grn_truth.csv", index=False)

    # ---- choose KO genes (the TFs) ----
    ko_genes = [gene_names[t] for t in tf_idx[:args.n_kos]]
    print(f"[run] KO genes: {ko_genes}")

    rows = []
    roc_curves = {}
    for i, ko in enumerate(ko_genes):
        ko_idx = gene_names.index(ko)
        # ground-truth children of ko
        children = np.where(truth[ko_idx, :])[0]
        labels = np.zeros(args.n_genes, dtype=int)
        labels[children] = 1
        print(f"\n[{i+1}/{len(ko_genes)}] KO={ko}  true_children={labels.sum()}")

        # GenKI
        try:
            sc_genki = genki_ko_score(adata, ko, args.epochs, args.lr,
                                      args.beta, args.seed, grn_dir)
            a_genki, p_genki = score_recovery(sc_genki, labels, gene_names)
        except Exception as e:
            print(f"  [GenKI] FAILED: {e}")
            a_genki, p_genki = float("nan"), float("nan")

        # scTenifoldKnk
        if args.skip_knk:
            a_knk, p_knk = float("nan"), float("nan")
        else:
            try:
                sc_knk = sctenifold_ko_score(adata_df, ko)
                a_knk, p_knk = score_recovery(sc_knk, labels, gene_names)
            except Exception as e:
                print(f"  [Knk] FAILED: {e}")
                a_knk, p_knk = float("nan"), float("nan")

        # baselines
        rng = np.random.default_rng(args.seed + i)
        sc_rand = dict(zip(gene_names, rng.uniform(0, 1, args.n_genes)))
        a_rand, p_rand = score_recovery(sc_rand, labels, gene_names)

        corr = np.abs(np.corrcoef(Xc[:, ko_idx], Xc, rowvar=False)[0, :])
        sc_corr = dict(zip(gene_names, corr))
        a_corr, p_corr = score_recovery(sc_corr, labels, gene_names)

        rows.append({"KO": ko, "n_true_children": int(labels.sum()),
                     "AUROC_GenKI": a_genki, "AUPRC_GenKI": p_genki,
                     "AUROC_Knk": a_knk, "AUPRC_Knk": p_knk,
                     "AUROC_rand": a_rand, "AUPRC_rand": p_rand,
                     "AUROC_corr": a_corr, "AUPRC_corr": p_corr})
        print(f"  GenKI AUROC={a_genki:.3f} AP={p_genki:.3f} | "
              f"Knk AUROC={a_knk:.3f} AP={p_knk:.3f} | "
              f"rand AUROC={a_rand:.3f} | corr AUROC={a_corr:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "benchmark_metrics.csv", index=False)
    print(f"\n[OK] {rstr(out_dir / 'benchmark_metrics.csv')}")

    # ---- summary ----
    summary = ["GenKI vs scTenifoldKnk — SERGIO-style benchmark (ground-truth GRN)",
               "=" * 60, ""]
    for col, label in [("AUROC_GenKI", "GenKI"), ("AUROC_Knk", "scTenifoldKnk"),
                       ("AUROC_corr", "Pearson-corr baseline"),
                       ("AUROC_rand", "Random baseline")]:
        v = res[col].dropna()
        summary.append(f"{label:22s} AUROC = {v.mean():.3f} +/- {v.std():.3f}  (n={len(v)})")
    summary.append("")
    for col, label in [("AUPRC_GenKI", "GenKI"), ("AUPRC_Knk", "scTenifoldKnk"),
                       ("AUPRC_corr", "Pearson-corr baseline"),
                       ("AUPRC_rand", "Random baseline")]:
        v = res[col].dropna()
        summary.append(f"{label:22s} AUPRC  = {v.mean():.3f} +/- {v.std():.3f}  (n={len(v)})")
    summary = "\n".join(summary) + "\n"
    (out_dir / "benchmark_summary.txt").write_text(summary)
    print("\n" + summary)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        methods = [("AUROC_GenKI", "GenKI", "C0"),
                   ("AUROC_Knk", "scTenifoldKnk", "C1"),
                   ("AUROC_corr", "Correlation", "C2"),
                   ("AUROC_rand", "Random", "C3")]
        data_auroc = [res[c].dropna().values for c, _, _ in methods]
        axes[0].boxplot(data_auroc, tick_labels=[l for _, l, _ in methods], showmeans=True)
        axes[0].axhline(0.5, color="gray", ls="--", lw=0.7)
        axes[0].set_ylabel("AUROC (recover true children)")
        axes[0].set_title("Ground-truth child recovery")

        data_auprc = [res[c.replace("AUROC", "AUPRC")].dropna().values for c, _, _ in methods]
        axes[1].boxplot(data_auprc, tick_labels=[l for _, l, _ in methods], showmeans=True)
        axes[1].set_ylabel("Average precision")
        axes[1].set_title("Ground-truth child recovery (AP)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_sergio_benchmark.pdf")
        print(f"[OK] {rstr(out_dir / 'fig_sergio_benchmark.pdf')}")
    except Exception as e:
        print(f"[WARN] figure not generated: {e}")


if __name__ == "__main__":
    main()
