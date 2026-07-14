"""Extract MAPT survival data and draw KM curve for Figure 4 panel f."""
import pandas as pd, numpy as np, os
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FIG4 = ROOT / "output/Figures_final/Figure4/data"

# 1. Load NULISA plasma protein (MAPT)
nulisa = pd.read_csv(ROOT / "data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv", low_memory=False)
plasma = nulisa[
    (nulisa['SampleMatrixType']=='PLASMA') &
    (nulisa['VISCODE']=='bl') &
    (nulisa['SampleQC']=='passed') &
    (nulisa['RID'].notna()) &
    (nulisa['Target']=='MAPT')
].copy()
plasma['RID'] = plasma['RID'].astype(int)
mapt_expr = plasma[['RID','NPQ']].drop_duplicates('RID')
print(f"MAPT plasma baseline: {len(mapt_expr)} samples")

# 2. Load DXSUM for diagnosis + time
dxsum = pd.read_csv(ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")
print(f"DXSUM: {dxsum.shape}, cols: {list(dxsum.columns[:10])}")

# 3. Build survival: event = progression to AD dementia, time = months from bl
# DXSUM has DXCHANGE, VISCODE2, etc. Need to figure out time-to-event.
# Check available columns for survival
survival_cols = [c for c in dxsum.columns if any(k in c.upper() for k in ['VISIT','MONTH','YEAR','DAY','TIME','CHANG','PROG','EVENT'])]
print(f"Survival-related cols: {survival_cols[:15]}")
print()

# The survival script builds time from visits. Let me check what it does.
print("DXCHANGE unique:", sorted(dxsum.get('DXCHANGE', pd.Series()).dropna().unique())[:15])
print("VISCODE2 sample:", sorted(dxsum.get('VISCODE2', pd.Series()).dropna().unique())[:10] if 'VISCODE2' in dxsum.columns else "no VISCODE2")
