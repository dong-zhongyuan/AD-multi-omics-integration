#!/usr/bin/env python3
"""
Aging pseudotime on TMS brain single-cell data + validated-target expression dynamics.

Computes a continuous aging pseudotime from Tabula Muris Senis brain scRNA
(5,000 brain cells across ages 3m / 18m / 24m) via diffusion map + DPT, rooted
at the 3-month-old centroid. Aging is the strongest non-genetic AD risk factor,
so the pseudotime axis represents "where in the aging trajectory each cell sits".

Then maps the human validated target gene set to mouse 1:1 orthologs and projects
their expression onto the continuous pseudotime, producing per-bin mean expression
for the downstream 3D surface heatmap (Figure 4 panel h).

Inputs:
  processed-data/step4_single_cell_tms/TMS_expression_matrix_for_step4.h5ad
  data/metadata/orthologs_mgi_1to1.csv      (mouse_symbol -> human_symbol, 17,609 pairs)
  target gene set (Cox + AUC validated human targets)

Outputs (output/step5_clinical_validation/expression_dynamics/):
  cell_pseudotime.csv            cell_id, age, cell_type, subtissue, pseudotime
  pseudotime_expression_matrix.csv   gene(human) x pseudotime_bin (mean z-expr)
  pseudotime_bins.csv                bin metadata (n cells, dominant age, pt midpoint)
  pseudotime_expression_long.csv     long form
  targets_used.csv                   human target + mouse ortholog + source
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMS_FILE = os.path.join(PROJECT, "processed-data/step4_single_cell_tms/TMS_expression_matrix_for_step4.h5ad")
ORTHO_FILE = os.path.join(PROJECT, "data/metadata/orthologs_mgi_1to1.csv")
OUT_DIR = os.path.join(PROJECT, "output/step5_clinical_validation/expression_dynamics")
os.makedirs(OUT_DIR, exist_ok=True)

N_BINS = 30
N_HVG = 2500

AGE_ORDER = ["3m", "18m", "24m"]

# ---------------------------------------------------------------- target set
def load_targets():
    sources = {}
    df = pd.read_csv(os.path.join(PROJECT, "data/figure_data/Figure4/fig4f_cox_proteomics.csv"))
    for g in df["gene"]: sources.setdefault(g, []).append("cox_prot")
    df = pd.read_csv(os.path.join(PROJECT, "data/figure_data/Figure4/fig4c_top_auc.csv"))
    for g in df[df["omics"] == "Proteomics"]["gene"]: sources.setdefault(g, []).append("auc_prot")
    df = pd.read_csv(os.path.join(PROJECT, "output/step5_diagnostic_performance/single_gene_auc_transcriptomics.csv"))
    for g in df.sort_values("single_auc", ascending=False).head(12)["gene"]:
        sources.setdefault(g, []).append("auc_trans")
    for g in ["CSF3R", "MMP9"]:
        sources.setdefault(g, []).append("cox_trans_paper_target")
    rows = [{"gene_human": g, "sources": "|".join(sorted(set(s)))} for g, s in sources.items()]
    return pd.DataFrame(rows).sort_values("gene_human").reset_index(drop=True)

def main():
    sc.settings.verbosity = 0

    print("Loading TMS brain scRNA...")
    adata = sc.read_h5ad(TMS_FILE)
    adata = adata[adata.obs["tissue"] == "Brain"].copy()
    # clean subtissue trailing whitespace
    adata.obs["subtissue"] = adata.obs["subtissue"].astype(str).str.strip()
    adata.obs["age"] = pd.Categorical(adata.obs["age"].astype(str), categories=AGE_ORDER, ordered=True)
    print(f"  brain cells: {adata.n_obs}, ages: {adata.obs['age'].value_counts().to_dict()}")

    # standard scRNA preprocessing
    sc.pp.filter_genes(adata, min_cells=10)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3",
                                layer="counts" if "counts" in adata.layers else None)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)

    # diffusion map + DPT
    print("Computing PCA + neighbors + diffusion map + DPT...")
    sc.tl.pca(adata_hvg, n_comps=30, svd_solver="randomized")
    sc.pp.neighbors(adata_hvg, n_neighbors=20, n_pcs=30)
    sc.tl.diffmap(adata_hvg, n_comps=15)

    # root = 3m centroid in diffusion space
    young_mask = (adata_hvg.obs["age"] == "3m").values
    young_centroid = adata_hvg.obsm["X_diffmap"][young_mask].mean(axis=0)
    dists = np.linalg.norm(adata_hvg.obsm["X_diffmap"] - young_centroid, axis=1)
    adata_hvg.uns["iroot"] = int(np.argmin(dists))
    sc.tl.dpt(adata_hvg, n_branchings=0, n_dcs=10)

    adata.obs["pseudotime"] = adata_hvg.obs["dpt_pseudotime"].values
    # If root picked at 3m but pseudotime still anti-correlates with age (3m ends
    # up at high DPT), flip sign so 3m=young (low) -> 24m=old (high).
    age_num = adata.obs["age"].map({"3m":0, "18m":1, "24m":2})
    from scipy.stats import spearmanr
    rho_check, _ = spearmanr(age_num, adata.obs["pseudotime"].replace([np.inf,-np.inf], np.nan))
    if rho_check < 0:
        adata.obs["pseudotime"] = -adata.obs["pseudotime"]
        print(f"  (flipped pseudotime sign; raw Spearman was {rho_check:.3f})")
    # drop undefined DPT (disconnected manifold components)
    pt = adata.obs["pseudotime"].replace([np.inf, -np.inf], np.nan)
    n_drop = pt.isna().sum()
    if n_drop:
        print(f"  {n_drop} cells had undefined DPT (dropped)")
        adata = adata[~pt.isna()].copy()
        pt = pt.dropna()
    adata.obs["pseudotime"] = pt

    # ---- sanity check: pseudotime vs age ----
    print("\nPseudotime by age (sanity check; expect monotonic increase):")
    age_pt = adata.obs.groupby("age", observed=True)["pseudotime"].agg(["mean", "median", "count"])
    print(age_pt.round(3).to_string())
    from scipy.stats import spearmanr
    age_num = adata.obs["age"].map({"3m":0, "18m":1, "24m":2})
    rho, p = spearmanr(age_num, adata.obs["pseudotime"])
    print(f"Spearman(age, pseudotime) = {rho:.3f} (p={p:.3g})")

    # ---- map human targets -> mouse orthologs ----
    print("\nMapping targets to mouse orthologs...")
    targets = load_targets()
    ortho = pd.read_csv(ORTHO_FILE)
    targets = targets.merge(ortho, left_on="gene_human", right_on="human_symbol", how="left")
    targets = targets.rename(columns={"mouse_symbol": "gene_mouse"})
    # Case mismatch: ortholog mouse_symbol is Title-case (Mapt) but TMS var_names
    # are UPPER (MAPT). Normalize for lookup, then keep the UPPER form that matches adata.
    tms_vars_upper = {v.upper(): v for v in adata.var_names}
    targets["gene_mouse_tms"] = targets["gene_mouse"].apply(
        lambda m: tms_vars_upper.get(m.upper()) if isinstance(m, str) else None)
    present_mouse = targets[targets["gene_mouse_tms"].notna()].copy()
    dropped = targets[targets["gene_mouse_tms"].isna()]
    n_no_ortho = targets["gene_mouse"].isna().sum()
    print(f"  {len(targets)} human targets; {n_no_ortho} have no 1:1 mouse ortholog; "
          f"{len(dropped)-n_no_ortho} ortholog absent from TMS brain; "
          f"{len(present_mouse)} usable")
    if len(dropped):
        print(f"  dropped: {dropped['gene_human'].tolist()}")

    # ---- extract ortholog expression on log-normalized scale ----
    sub = adata[:, present_mouse["gene_mouse_tms"].tolist()].copy()
    X_dense = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    expr = pd.DataFrame(X_dense, index=sub.obs_names, columns=present_mouse["gene_human"].tolist())
    expr["pseudotime"] = adata.obs["pseudotime"].values
    expr["age"] = adata.obs["age"].astype(str).values

    # z-score each gene across cells
    mu = expr[present_mouse["gene_human"]].mean()
    sd = expr[present_mouse["gene_human"]].std().replace(0, np.nan)
    expr_z = expr[present_mouse["gene_human"]].sub(mu).div(sd)
    expr_z["pseudotime"] = expr["pseudotime"].values
    expr_z["age"] = expr["age"].values

    # bin pseudotime
    pt_min, pt_max = expr["pseudotime"].min(), expr["pseudotime"].max()
    bins = np.linspace(pt_min, pt_max, N_BINS + 1)
    expr["pt_bin"] = pd.cut(expr["pseudotime"], bins=bins, labels=False, include_lowest=True)
    expr_z["pt_bin"] = expr["pt_bin"].values

    mat = expr_z.groupby("pt_bin")[present_mouse["gene_human"]].mean()
    bin_meta = expr.groupby("pt_bin").agg(
        n_cells=("age", "size"),
        age_mode=("age", lambda s: s.value_counts().idxmax() if len(s) else "NA"),
        pt_mid=("pseudotime", "mean"),
    )
    long = expr_z.melt(id_vars=["pt_bin", "age"], var_name="gene", value_name="z_expr").dropna(subset=["z_expr"])

    # save
    adata.obs[["age", "cell_ontology_class", "subtissue", "pseudotime"]].to_csv(
        os.path.join(OUT_DIR, "cell_pseudotime.csv"))
    mat.T.to_csv(os.path.join(OUT_DIR, "pseudotime_expression_matrix.csv"))
    bin_meta.to_csv(os.path.join(OUT_DIR, "pseudotime_bins.csv"))
    long.to_csv(os.path.join(OUT_DIR, "pseudotime_expression_long.csv"), index=False)
    present_mouse.to_csv(os.path.join(OUT_DIR, "targets_used.csv"), index=False)
    print(f"\nSaved to {OUT_DIR}:")
    print(f"  cell_pseudotime.csv              {len(adata.obs)} cells")
    print(f"  pseudotime_expression_matrix.csv     {mat.shape[1]} genes x {mat.shape[0]} bins")
    print(f"  pseudotime_bins.csv               bin metadata")
    print(f"  pseudotime_expression_long.csv    {len(long)} rows")
    print(f"  targets_used.csv                  {len(present_mouse)} genes (with ortholog)")
    print(f"\nPseudotime range: [{pt_min:.3f}, {pt_max:.3f}], {N_BINS} bins")

if __name__ == "__main__":
    main()
