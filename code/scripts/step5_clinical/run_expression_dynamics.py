#!/usr/bin/env python3
"""
Diffusion pseudotime of ADNI subjects + target expression dynamics.

Computes a REAL continuous pseudotime for each ADNI subject based on
transcriptome-wide similarity (diffusion map + diffusion pseudotime, DPT),
using scanpy. The root is set to the CN-subject centroid in diffusion space,
so pseudotime increases along the CN -> MCI -> AD direction.

Then projects the validated target gene set onto this continuous axis,
producing per-pseudotime-bin mean expression for the downstream 3D surface
heatmap (Figure 4 panel h).

Inputs:
  data/survival/ADNI_Gene_Expression_Profile.csv   (49k probes x 745 samples, log2)
  data/blood-transcription-protein/DXSUM_17Apr2026.csv  (PTID + DIAGNOSIS)
  target gene set (Cox + AUC validated targets)

Outputs (output/step5_clinical_validation/expression_dynamics/):
  subject_pseudotime.csv        PTID, DIAGNOSIS, stage, pseudotime
  pseudotime_expression_matrix.csv  gene x pseudotime_bin (mean z-expr)
  pseudotime_expression_long.csv    long form for plotting
  targets_used.csv                 final target list with source annotation
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPR_FILE = os.path.join(PROJECT, "data", "survival", "ADNI_Gene_Expression_Profile.csv")
DXSUM_FILE = os.path.join(PROJECT, "data", "blood-transcription-protein", "DXSUM_17Apr2026.csv")
OUT_DIR = os.path.join(PROJECT, "output", "step5_clinical_validation", "expression_dynamics")
os.makedirs(OUT_DIR, exist_ok=True)

STAGE_MAP = {1: "CN", 2: "MCI", 3: "AD"}
N_BINS = 30                          # pseudotime binning for the surface mesh
N_HVG = 2000                         # highly-variable genes for diffusion map

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
    rows = [{"gene": g, "sources": "|".join(sorted(set(s)))} for g, s in sources.items()]
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)

# ---------------------------------------------------------- parse ADNI expr
def load_expression():
    """Return (gene x subject matrix, batch covariates per subject).

    Batch covariates (Phase, Affy Plate, RIN, Year) are parsed from the stacked
    header rows so they can be regressed out before pseudotime computation.
    """
    raw = pd.read_csv(EXPR_FILE, header=None, low_memory=False)
    ptids = raw.iloc[2, 3:].astype(str).tolist()
    data = raw.iloc[9:, 3:].reset_index(drop=True)
    symbols = raw.iloc[9:, 2].astype(str).reset_index(drop=True)
    data = data.apply(pd.to_numeric, errors="coerce")
    data.columns = ptids
    data["Symbol"] = symbols.values
    data = data[data["Symbol"].notna() & (data["Symbol"] != "") & (~data["Symbol"].str.startswith("AFFX"))]
    gmat = data.groupby("Symbol").median(numeric_only=True)
    # batch covariates (per subject)
    batch = pd.DataFrame({
        "PTID":     ptids,
        "Phase":    raw.iloc[0, 3:].astype(str).tolist(),
        "Plate":    raw.iloc[6, 3:].apply(pd.to_numeric, errors="coerce").tolist(),
        "RIN":      raw.iloc[5, 3:].apply(pd.to_numeric, errors="coerce").tolist(),
        "Year":     raw.iloc[7, 3:].apply(pd.to_numeric, errors="coerce").tolist(),
    })
    return gmat, batch

def main():
    sc.settings.verbosity = 0

    print("Loading target set...")
    targets = load_targets()
    print(f"  {len(targets)} candidate target genes")

    print("Loading ADNI expression (~30s, 222MB)...")
    gmat, batch = load_expression()
    print(f"  expression matrix: {gmat.shape[0]} genes x {gmat.shape[1]} subjects")

    print("Loading DXSUM diagnosis (baseline)...")
    dx = pd.read_csv(DXSUM_FILE, low_memory=False)
    dx = dx[dx["VISCODE2"].astype(str).str.lower() == "bl"]
    dx = dx[["PTID", "DIAGNOSIS"]].dropna()
    dx["PTID"] = dx["PTID"].astype(str)
    dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx.dropna()
    dx["stage"] = dx["DIAGNOSIS"].map(STAGE_MAP)
    dx = dx[dx["stage"].notna()]
    ptid2stage = dict(zip(dx["PTID"], dx["stage"]))
    ptid2dx = dict(zip(dx["PTID"], dx["DIAGNOSIS"]))

    # keep subjects with both expression and diagnosis
    common = [p for p in gmat.columns if p in ptid2stage]
    gmat = gmat[common]
    batch = batch[batch["PTID"].isin(common)].set_index("PTID").loc[common]
    print(f"  subjects with expr + dx: {len(common)} "
          f"({pd.Series([ptid2stage[p] for p in common]).value_counts().to_dict()})")

    # ---- build AnnData for scanpy (subjects as observations) ----
    X = gmat.T.values.astype(np.float32)
    adata = sc.AnnData(X)
    adata.obs_names = list(gmat.columns)
    adata.var_names = list(gmat.index)
    adata.obs["stage"] = [ptid2stage[p] for p in adata.obs_names]
    adata.obs["DIAGNOSIS"] = [ptid2dx[p] for p in adata.obs_names]
    # attach batch covariates
    adata.obs["Phase"] = batch["Phase"].values
    adata.obs["Plate"] = batch["Plate"].values
    adata.obs["RIN"]   = batch["RIN"].values
    adata.obs["Year"]  = batch["Year"].values

    # ---- regress out batch covariates (Phase, Plate, RIN, Year) per gene ----
    # Use scanpy's regress_out on the HVG subset below (after scaling).
    # First: HVG selection, then scale, then regress out technical covariates.
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)

    # regress out batch on the scaled HVG matrix (numeric covariates; Phase as dummy)
    print("Regressing out batch covariates (Phase, Plate, RIN, Year)...")
    # scanpy regress_out handles one variable at a time; encode Phase numerically
    adata_hvg.obs["Phase_num"] = pd.Categorical(adata_hvg.obs["Phase"]).codes.astype(float)
    for cov in ["Phase_num", "Plate", "RIN", "Year"]:
        adata_hvg.obs[cov] = adata_hvg.obs[cov].fillna(adata_hvg.obs[cov].mean())
        try:
            sc.pp.regress_out(adata_hvg, [cov])
        except Exception as e:
            print(f"  (regress_out {cov} skipped: {e})")
    # re-scale after regression
    sc.pp.scale(adata_hvg, max_value=10)

    # PCA -> diffusion map -> DPT
    print("Computing PCA + diffusion map + DPT...")
    sc.tl.pca(adata_hvg, n_comps=30, svd_solver="randomized")
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
    sc.tl.diffmap(adata_hvg, n_comps=15)

    # root = CN-subject centroid in diffusion space (DC1..DC15)
    cn_mask = (adata_hvg.obs["stage"] == "CN").values
    root_idx = np.argmax(np.all(
        adata_hvg.obsm["X_diffmap"][cn_mask] == adata_hvg.obsm["X_diffmap"][cn_mask].mean(axis=0),
        axis=1)) if cn_mask.sum() > 0 else 0
    # simpler & robust: pick the CN subject closest to CN centroid
    cn_centroid = adata_hvg.obsm["X_diffmap"][cn_mask].mean(axis=0)
    dists = np.linalg.norm(adata_hvg.obsm["X_diffmap"] - cn_centroid, axis=1)
    root_local = int(np.argmin(dists))
    adata_hvg.uns["iroot"] = root_local
    sc.tl.dpt(adata_hvg, n_branchings=0, n_dcs=15)

    # copy pseudotime back to full adata
    adata.obs["pseudotime"] = adata_hvg.obs.get("dpt_pseudotime",
                                                adata_hvg.obs.get("pseudotime")).values

    # fix infinite/NaN DPT values (disconnected components)
    pt = adata.obs["pseudotime"].replace([np.inf, -np.inf], np.nan)
    n_nan = pt.isna().sum()
    if n_nan:
        print(f"  {n_nan} subjects had undefined DPT (dropped)")
        adata = adata[~pt.isna()].copy()
        pt = pt.dropna()
    adata.obs["pseudotime"] = pt

    # ---- pseudotime sanity check: does it correlate with CN->MCI->AD? ----
    print("\nPseudotime by clinical stage (sanity check):")
    stage_pt = adata.obs.groupby("stage")["pseudotime"].agg(["mean", "median", "count"])
    print(stage_pt.round(3).to_string())

    # ---- project targets onto pseudotime bins ----
    targets_present = targets[targets["gene"].isin(adata.var_names)].copy()
    dropped = targets[~targets["gene"].isin(adata.var_names)]
    if len(dropped):
        print(f"\nWARNING: {len(dropped)} targets not on array, dropped: {dropped['gene'].tolist()}")

    sub = adata[:, targets_present["gene"]].copy()
    expr = pd.DataFrame(sub.X, index=sub.obs_names, columns=sub.var_names)
    expr["pseudotime"] = adata.obs["pseudotime"].values
    expr["stage"] = adata.obs["stage"].values

    # z-score each gene across subjects
    mu = expr[targets_present["gene"]].mean()
    sd = expr[targets_present["gene"]].std().replace(0, np.nan)
    expr_z = expr[targets_present["gene"]].sub(mu).div(sd)

    # bin pseudotime into N_BINS
    pt_min, pt_max = expr["pseudotime"].min(), expr["pseudotime"].max()
    bins = np.linspace(pt_min, pt_max, N_BINS + 1)
    expr["pt_bin"] = pd.cut(expr["pseudotime"], bins=bins, labels=False, include_lowest=True)
    expr_z["pt_bin"] = expr["pt_bin"].values
    expr_z["stage"] = expr["stage"].values

    # gene x pt_bin mean z-expression
    mat = expr_z.groupby("pt_bin")[targets_present["gene"]].mean()
    # also count + dominant stage per bin (for context)
    bin_meta = expr.groupby("pt_bin").agg(
        n=("stage", "size"),
        stage_mode=("stage", lambda s: s.value_counts().idxmax() if len(s) else "NA"),
        pt_mid=("pseudotime", "mean"),
    )

    # long form
    long = expr_z.melt(id_vars=["pt_bin", "stage"], var_name="gene", value_name="z_expr")
    long = long.dropna(subset=["z_expr"])

    # save
    adata.obs[["stage", "DIAGNOSIS", "pseudotime"]].to_csv(
        os.path.join(OUT_DIR, "subject_pseudotime.csv"))
    mat.T.to_csv(os.path.join(OUT_DIR, "pseudotime_expression_matrix.csv"))   # gene x bin
    bin_meta.to_csv(os.path.join(OUT_DIR, "pseudotime_bins.csv"))
    long.to_csv(os.path.join(OUT_DIR, "pseudotime_expression_long.csv"), index=False)
    targets_present.to_csv(os.path.join(OUT_DIR, "targets_used.csv"), index=False)

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  subject_pseudotime.csv       {len(adata.obs)} subjects")
    print(f"  pseudotime_expression_matrix.csv  {mat.shape[1]} genes x {mat.shape[0]} bins")
    print(f"  pseudotime_bins.csv          bin metadata (n, dominant stage, pt midpoint)")
    print(f"  pseudotime_expression_long.csv    {len(long)} rows")
    print(f"  targets_used.csv             {len(targets_present)} genes")
    print(f"\nPseudotime range: [{pt_min:.3f}, {pt_max:.3f}], {N_BINS} bins")

if __name__ == "__main__":
    main()
