"""Extract MAPT survival data and draw KM curve for Figure 4 panel f.
Event = progression to AD dementia (DIAGNOSIS=3).
Time = months from baseline to first AD diagnosis (or last follow-up censored).
Group = MAPT NPQ high (>= median) vs low (< median).
"""
import pandas as pd, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_FIG4 = ROOT / "output/Figures_final/Figure4/output"
DATA_FIG4 = ROOT / "output/Figures_final/Figure4/data"
os.makedirs(OUT_FIG4, exist_ok=True)

# 1. MAPT plasma expression at baseline
nulisa = pd.read_csv(ROOT / "data/blood-transcription-protein/BSHRI_PLA_CSF_NULISA_CNS_16Apr2026.csv", low_memory=False)
plasma = nulisa[
    (nulisa['SampleMatrixType']=='PLASMA') & (nulisa['VISCODE']=='bl') &
    (nulisa['SampleQC']=='passed') & (nulisa['RID'].notna()) & (nulisa['Target']=='MAPT')
].copy()
plasma['RID'] = plasma['RID'].astype(int)
mapt = plasma[['RID','NPQ']].drop_duplicates('RID').rename(columns={'NPQ':'MAPT'})

# 2. DXSUM: build time-to-AD-progression
dx = pd.read_csv(ROOT / "data/blood-transcription-protein/DXSUM_17Apr2026.csv")
# Parse VISCODE2 to months (bl=0, m06=6, m12=12, ...)
def viscode_to_months(v):
    if pd.isna(v): return np.nan
    v = str(v).strip().lower()
    if v == 'bl': return 0
    if v.startswith('m'):
        try: return int(v[1:])
        except: return np.nan
    return np.nan

dx['months'] = dx['VISCODE2'].apply(viscode_to_months)
dx['RID'] = pd.to_numeric(dx['RID'], errors='coerce')
dx['DIAGNOSIS'] = pd.to_numeric(dx['DIAGNOSIS'], errors='coerce')

# Baseline diagnosis: include MCI + NL (not yet AD)
bl_dx = dx[dx['VISCODE2']=='bl'][['RID','DIAGNOSIS']].drop_duplicates('RID')
bl_dx = bl_dx[bl_dx['DIAGNOSIS'].isin([1,2])]  # NL or MCI at baseline
bl_dx = bl_dx.rename(columns={'DIAGNOSIS':'bl_dx'})

# For each subject: find first AD diagnosis (DIAGNOSIS=3), time = months
ad_first = dx[dx['DIAGNOSIS']==3].sort_values('months').drop_duplicates('RID')[['RID','months']]
ad_first = ad_first.rename(columns={'months':'time_event'})

# Last follow-up for censored
last_fu = dx.sort_values('months').drop_duplicates('RID', keep='last')[['RID','months']]
last_fu = last_fu.rename(columns={'months':'time_censor'})

# Merge
surv = bl_dx.merge(ad_first, on='RID', how='left').merge(last_fu, on='RID', how='left')
surv['event'] = surv['time_event'].notna().astype(int)
surv['time'] = surv['time_event'].fillna(surv['time_censor'])
surv = surv.dropna(subset=['time'])
surv = surv[surv['time'] > 0]  # exclude 0-month (baseline)

# Merge with MAPT
merged = surv.merge(mapt, on='RID', how='inner')
print(f"Merged: {len(merged)} subjects with MAPT + survival")
print(f"Events (progressed to AD): {merged['event'].sum()}")

# Split by MAPT median
median_mapt = merged['MAPT'].median()
merged['group'] = np.where(merged['MAPT'] >= median_mapt, 'High MAPT', 'Low MAPT')
print(f"Median MAPT NPQ: {median_mapt:.2f}")
print(f"High MAPT: {(merged['group']=='High MAPT').sum()}, Low MAPT: {(merged['group']=='Low MAPT').sum()}")

# Save data
merged[['RID','MAPT','time','event','group']].to_csv(DATA_FIG4 / "fig4f_mapt_km.csv", index=False)
print(f"Saved fig4f_mapt_km.csv")

# 3. Draw KM curve
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

fig_w, fig_h = 3.0, 3.0
FS = round(5.6 * 3.0); FSL = round(6.4 * 3.0)
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': FS, 'axes.linewidth': 0.8,
    'axes.labelsize': FSL, 'axes.labelweight': 'bold',
    'xtick.labelsize': FS, 'ytick.labelsize': FS,
    'figure.constrained_layout.use': True,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
kmf = KaplanMeierFitter()
COLORS = {'High MAPT': '#D55E00', 'Low MAPT': '#0072B2'}

for grp in ['High MAPT', 'Low MAPT']:
    sub = merged[merged['group']==grp]
    kmf.fit(sub['time'], sub['event'], label=grp)
    kmf.plot_survival_function(ax=ax, color=COLORS[grp], ci_show=True, linewidth=2,
                               alpha=0.8, show_censors=True, censor_styles={'ms':5})

# Log-rank test
g1 = merged[merged['group']=='High MAPT']
g2 = merged[merged['group']=='Low MAPT']
lr = logrank_test(g1['time'], g2['time'], g1['event'], g2['event'])
ax.text(0.05, 0.05, f'log-rank p = {lr.p_value:.4f}', transform=ax.transAxes,
        fontsize=FS-4, fontweight='bold', color='#333')

ax.set_xlabel('Months from baseline', fontsize=FS)
ax.set_ylabel('AD-free probability', fontsize=FS)
ax.legend(frameon=False, loc='upper right', prop={'size': FS-4})
ax.set_ylim(0, 1.05)
ax.grid(linestyle=':', alpha=0.4)

for ext in ['png','pdf','svg']:
    plt.savefig(OUT_FIG4 / f'f.{ext}', dpi=300)
plt.close()
print(f"Saved f.png/pdf/svg (KM curve for MAPT)")
