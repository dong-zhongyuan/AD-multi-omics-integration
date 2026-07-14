# prep_fig3c_confidence.py — extract per-target confidence data for Figure 3 panel c.
# Reads all GenKI *_negative_controls.csv files (forward + reverse, proteomics + transcriptomics)
# and produces a single tidy CSV used by the confidence-scatter panel.
#
# Output columns: omics, direction, ko_gene, target_gene, target_expression,
#                  cohens_d, p_value, significant
import os
import glob
import pandas as pd

ROOT = ros.path.join(str(PROJECT_ROOT), "output/step4_virtual_knockout")
OUT_DIR = ros.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/data")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = [
    ("Proteomics",     "Forward", os.path.join(ROOT, "GenKI_NO3",         "proteomics_*_negative_controls.csv")),
    ("Transcriptomics","Forward", os.path.join(ROOT, "GenKI_NO3",         "transcriptomics_*_negative_controls.csv")),
    ("Proteomics",     "Reverse", os.path.join(ROOT, "GenKI_NO3_reverse", "proteomics_*_negative_controls.csv")),
    ("Transcriptomics","Reverse", os.path.join(ROOT, "GenKI_NO3_reverse", "transcriptomics_*_negative_controls.csv")),
]

KEEP = ["target_gene", "target_expression", "cohens_d", "p_value", "significant", "ko_gene"]
frames = []
for omics, direction, pattern in SOURCES:
    files = glob.glob(pattern)
    for f in files:
        df = pd.read_csv(f)
        # some files use slightly different casing; keep what exists
        cols = [c for c in KEEP if c in df.columns]
        df = df[cols].copy()
        df["omics"] = omics
        df["direction"] = direction
        frames.append(df)

big = pd.concat(frames, ignore_index=True)
# normalize boolean / numeric
big["significant"] = big["significant"].astype(str).str.lower().isin(["true", "1"])
for c in ["target_expression", "cohens_d", "p_value"]:
    big[c] = pd.to_numeric(big[c], errors="coerce")
big = big.dropna(subset=["target_expression", "cohens_d", "p_value"])
# clip p_value to avoid log10(0)
big["p_value"] = big["p_value"].clip(lower=1e-300)

big = big[["omics", "direction", "ko_gene", "target_gene",
           "target_expression", "cohens_d", "p_value", "significant"]]

out = os.path.join(OUT_DIR, "fig3e_confidence.csv")
big.to_csv(out, index=False)
print(f"Saved {out}")
print(f"Rows: {len(big)}")
print(big.groupby(["omics", "direction"]).size())
