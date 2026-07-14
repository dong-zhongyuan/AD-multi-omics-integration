"""Extract MMP9 and CXCR2 survival data for Figure 5 KM curves.
Adapts build_mapt_km.py but uses ADNI gene expression microarray (transcriptomics)
instead of NULISA plasma protein.
Output: fig5e_mmp9_km.csv, fig5i_cxcr2_km.csv (subject-level: RID, gene_expr, time, event, group)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FIG5 = ROOT / "output/Figures_final/Figure5/data"

# 1. ADNI gene expression: extract MMP9 + CXCR2 baseline expression per subject
# File has 9-row stacked header; row 2 = SubjectID (PTID), row 9 header = Symbol
raw = pd.read_csv(ROOT / "data/survival/ADNI_Gene_Expression_Profile.csv",
                  header=None, low_memory=False)
ptids = raw.iloc[2, 3:].astype(str).tolist()
visits = raw.iloc[1, 3:].astype(str).tolist()
symbols = raw.iloc[9:, 2].astype(str).reset_index(drop=True)
data = raw.iloc[9:, 3:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
data.columns = ptids
data["Symbol"] = symbols

# keep only baseline visit columns (col indices 0..N-2 are subjects, last col is Symbol)
bl_mask = np.array([str(v).strip().lower() == "bl" for v in visits])
# data has N subject columns + 1 Symbol column; bl_mask applies to subject cols only
data_bl = data.iloc[:, list(np.where(bl_mask)[0]) + [data.shape[1] - 1]]
bl_ptids = [ptids[i] for i in np.where(bl_mask)[0]]
data_bl.columns = list(bl_ptids) + ["Symbol"]

for gene in ["MMP9", "CXCR2"]:
    g = data_bl[data_bl["Symbol"] == gene].select_dtypes(include=[np.number])
    if len(g) == 0:
        print(f"WARNING: {gene} not found in expression data")
        continue
    # collapse multiple probes (median)
    g_expr = g.median().rename(gene)
    print(f"{gene}: {len(g)} probes, {g_expr.notna().sum()} subjects with expression")

# 2. DXSUM: build time-to-AD-progression
dx = pd.read_csv(ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")

def viscode_to_months(v):
    if pd.isna(v): return np.nan
    v = str(v).strip().lower()
    if v == "bl": return 0
    if v.startswith("m"):
        try: return int(v[1:])
        except: return np.nan
    return np.nan

dx["months"] = dx["VISCODE2"].apply(viscode_to_months)
dx["PTID"] = dx["PTID"].astype(str)
dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")

# baseline NL or MCI
bl_dx = dx[dx["VISCODE2"] == "bl"][["PTID", "DIAGNOSIS"]].drop_duplicates("PTID")
bl_dx = bl_dx[bl_dx["DIAGNOSIS"].isin([1, 2])].rename(columns={"DIAGNOSIS": "bl_dx"})

# first AD diagnosis
ad_first = dx[dx["DIAGNOSIS"] == 3].sort_values("months").drop_duplicates("PTID")[["PTID", "months"]]
ad_first = ad_first.rename(columns={"months": "time_event"})
# last follow-up
last_fu = dx.sort_values("months").drop_duplicates("PTID", keep="last")[["PTID", "months"]]
last_fu = last_fu.rename(columns={"months": "time_censor"})

surv = bl_dx.merge(ad_first, on="PTID", how="left").merge(last_fu, on="PTID", how="left")
surv["event"] = surv["time_event"].notna().astype(int)
surv["time"] = surv["time_event"].fillna(surv["time_censor"])
surv = surv.dropna(subset=["time"])
surv = surv[surv["time"] > 0]
print(f"\nSurvival subjects (NL/MCI baseline): {len(surv)}, events: {surv['event'].sum()}")

# 3. Merge expression with survival for each gene, save KM data
for gene in ["MMP9", "CXCR2"]:
    g = data_bl[data_bl["Symbol"] == gene].select_dtypes(include=[np.number])
    if len(g) == 0: continue
    g_expr = g.median().rename(gene)
    g_df = g_expr.reset_index()
    g_df.columns = ["PTID", gene]
    merged = surv.merge(g_df, on="PTID", how="inner")
    merged = merged.dropna(subset=[gene])
    med = merged[gene].median()
    merged["group"] = np.where(merged[gene] >= med, f"High {gene}", f"Low {gene}")
    out_name = f"fig5{'e' if gene=='MMP9' else 'i'}_{gene.lower()}_km.csv"
    merged[["PTID", gene, "time", "event", "group"]].to_csv(DATA_FIG5 / out_name, index=False)
    print(f"\n{gene}: {len(merged)} subjects (High: {(merged['group'].str.startswith('High')).sum()}, "
          f"Low: {(merged['group'].str.startswith('Low')).sum()}), events: {merged['event'].sum()}")
    print(f"  median expr: {med:.3f}, saved {out_name}")
