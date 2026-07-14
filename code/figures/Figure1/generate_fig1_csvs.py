#!/usr/bin/env python3
"""Generate all Figure 1 CSV data files from pipeline output.

Copies FULL CSV files (no subsetting) and writes them to the Figure1/data
directory, renamed to match panel letters.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # D:\AD-Multi-Omics-Integration
OUT = ROOT / "output"
FIG1 = OUT / "Figures_final" / "Figure1"
DATA = FIG1 / "data"
DATA.mkdir(parents=True, exist_ok=True)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Panel (b) - UMAP data
# ---------------------------------------------------------------------------
banner("Panel (b) - UMAP")
src = OUT / "figures" / "umap_genotype_demultiplex.csv"
df_b = pd.read_csv(src)
df_b.to_csv(DATA / "fig1b_umap.csv", index=False)
print(f"rows={len(df_b)}, cols={len(df_b.columns)}")
print(f"columns: {list(df_b.columns)}")
print(df_b.head(2).to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (c) - Training convergence (3 omics) merged
# ---------------------------------------------------------------------------
banner("Panel (c) - Training convergence")
frames = []
omics_map = {
    "step1_world_model_transcriptomics_no_pca": "Transcriptomics",
    "step1_world_model_proteomics": "Proteomics",
    "step1_world_model_metabolomics": "Metabolomics",
}
for step_dir, omics in omics_map.items():
    p = OUT / step_dir / "train_log.csv"
    d = pd.read_csv(p)
    d.insert(1, "omics", omics)
    # Reorder to target schema
    target_cols = [
        "epoch", "omics", "train_total", "train_ot", "train_reg",
        "val_total", "val_ot", "val_reg", "sec",
    ]
    d = d[target_cols]
    frames.append(d)
    print(f"  {omics}: {len(d)} rows from {p.name}")
df_c = pd.concat(frames, ignore_index=True)
df_c.to_csv(DATA / "fig1c_training.csv", index=False)
print(f"merged rows={len(df_c)}, cols={len(df_c.columns)}")
print(df_c.head(2).to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (d) - Fair mapping benchmark (4 methods) + ot_distance_mean/sd
# ---------------------------------------------------------------------------
banner("Panel (d) - Fair mapping benchmark")
# Pivot the long-format results CSV to one row per method
res = pd.read_csv(OUT / "benchmark_world_model_advantages" / "benchmark_advantage_results.csv")
# Keep only group A fair mapping rows
res_a = res[res["group"] == "A_fair_mapping"].copy()
pivoted = (
    res_a.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
    .reset_index()
)
pivoted.columns.name = None
print("pivoted from results.csv:")
print(pivoted.to_string(index=False))

# Pull ot_distance_mean / ot_distance_sd from the summary JSON (group_A)
summary_path = OUT / "benchmark_world_model_advantages" / "benchmark_advantage_summary.json"
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)
ot_lookup = {}
for entry in summary["group_A_fair_mapping"]:
    ot_lookup[entry["method"]] = (
        entry.get("ot_distance_mean"),
        entry.get("ot_distance_sd"),
    )
print("ot_lookup:", ot_lookup)

# Add / overwrite ot_distance_mean and ot_distance_sd from JSON (authoritative)
pivoted["ot_distance_mean"] = pivoted["method"].map(lambda m: ot_lookup.get(m, (None, None))[0])
pivoted["ot_distance_sd"] = pivoted["method"].map(lambda m: ot_lookup.get(m, (None, None))[1])

# Order columns nicely: method first, then the metrics
preferred = [
    "method", "ot_distance_mean", "ot_distance_sd",
    "mmd_rbf", "corr_structure_mae",
]
ordered = [c for c in preferred if c in pivoted.columns]
extra = [c for c in pivoted.columns if c not in ordered]
df_d = pivoted[ordered + extra].copy()
df_d.to_csv(DATA / "fig1d_benchmark.csv", index=False)
print(f"rows={len(df_d)}, cols={len(df_d.columns)}")
print(f"columns: {list(df_d.columns)}")
print(df_d.to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (e) - Jacobian AUC
# ---------------------------------------------------------------------------
banner("Panel (e) - Jacobian AUC")
dim1 = summary["group_B_world_model_exclusive"]["dim1_jacobian_derivability"]
rows_e = [
    {
        "method": "NeuralODE",
        "auc": dim1["NeuralODE_jacobian_auc"],
        "note": "",
        "n_pairs": dim1["NeuralODE_n_pairs"],
    },
    {
        "method": "Ridge",
        "auc": dim1["Ridge_jacobian_auc"],
        "note": "",
        "n_pairs": dim1["Ridge_n_pairs"],
    },
    {
        "method": "DirectOT",
        "auc": np.nan,
        "note": "N/A (not differentiable, no Jacobian)",
        "n_pairs": 0,
    },
]
df_e = pd.DataFrame(rows_e, columns=["method", "auc", "note", "n_pairs"])
df_e.to_csv(DATA / "fig1e_jacobian_auc.csv", index=False)
print(df_e.to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (f) - Continuous trajectory
# ---------------------------------------------------------------------------
banner("Panel (f) - Continuous trajectory")
dim2 = summary["group_B_world_model_exclusive"]["dim2_continuous_trajectory"]
traj_times = dim2["traj_times"]
ot_to_blood = dim2["ot_to_blood"]
ot_to_brain = dim2["ot_to_brain"]
df_f = pd.DataFrame(
    {
        "t": traj_times,
        "ot_to_blood": ot_to_blood,
        "ot_to_brain": ot_to_brain,
    }
)
df_f.to_csv(DATA / "fig1f_trajectory.csv", index=False)
print(df_f.to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (g) - Edge confidence distributions per omics (ALL edges)
# ---------------------------------------------------------------------------
banner("Panel (g) - Edge confidence (ALL edges, merged)")
edge_sources = {
    "Transcriptomics": OUT / "step2_cross_tissue_causality" / "transcriptomics" / "filtered_edges" / "cross_tissue_edges.csv",
    "Proteomics": OUT / "step2_cross_tissue_causality" / "proteomics" / "filtered_edges" / "cross_tissue_edges.csv",
    "Metabolomics": OUT / "step2_cross_tissue_causality" / "metabolomics" / "filtered_edges" / "cross_tissue_edges.csv",
}
g_frames = []
for omics, p in edge_sources.items():
    d = pd.read_csv(p)
    d.insert(len(d.columns), "omics", omics)
    g_frames.append(d)
    print(f"  {omics}: {len(d)} rows")
df_g = pd.concat(g_frames, ignore_index=True)
# Target column order
target_g = [
    "source", "target", "weight", "strength",
    "confidence_stability", "confidence_snr", "confidence_consistency",
    "direction", "omics",
]
df_g = df_g[target_g]
df_g.to_csv(DATA / "fig1g_edge_confidence.csv", index=False)
print(f"merged rows={len(df_g)}, cols={len(df_g.columns)}")

# ---------------------------------------------------------------------------
# Panel (h) - Top proteomics edges (ALL 740 rows, ALL columns)
# ---------------------------------------------------------------------------
banner("Panel (h) - Top proteomics edges (ALL)")
src_h = edge_sources["Proteomics"]
df_h = pd.read_csv(src_h)
df_h.to_csv(DATA / "fig1h_proteomics_edges.csv", index=False)
print(f"rows={len(df_h)}, cols={len(df_h.columns)}")
print(f"columns: {list(df_h.columns)}")
print(df_h.head(2).to_string(index=False))

# ---------------------------------------------------------------------------
# Panel (i) - Reversibility + Network quality summary
# ---------------------------------------------------------------------------
banner("Panel (i) - Network summary")
dim3 = summary["group_B_world_model_exclusive"]["dim3_reversibility"]

# Compute per-omics summary stats from edge files
summary_stats = []
for omics, p in edge_sources.items():
    d = pd.read_csv(p)
    summary_stats.append(
        {
            "omics": omics,
            "n_edges": len(d),
            "mean_strength": float(d["strength"].mean()),
            "mean_stability": float(d["confidence_stability"].mean()),
            "mean_snr": float(d["confidence_snr"].mean()),
            "mean_consistency": float(d["confidence_consistency"].mean()),
        }
    )
stats_df = pd.DataFrame(summary_stats)

# Add reversibility columns (same value applies to the NeuralODE round-trip)
rev = {
    "roundtrip_corr": dim3["round_trip_corr"],
    "roundtrip_mse": dim3["round_trip_mse"],
    "roundtrip_mae": dim3["round_trip_mae"],
}
for k, v in rev.items():
    stats_df[k] = v

# Order columns
target_i = [
    "omics", "n_edges", "mean_strength", "mean_stability",
    "mean_snr", "mean_consistency",
    "roundtrip_corr", "roundtrip_mse", "roundtrip_mae",
]
df_i = stats_df[target_i].copy()
df_i.to_csv(DATA / "fig1i_network_summary.csv", index=False)
print(df_i.to_string(index=False))

banner("DONE")
print(f"All CSVs written to: {DATA}")
