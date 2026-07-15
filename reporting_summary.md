# Reporting Summary — Nature Machine Intelligence

## 1. Statistics

**Confirmed:** All items below are present in the figure legends, main text, or Methods section.

- **Sample size:** Each n is reported as a discrete number: 5xFAD scRNA-seq (10,000 cells × 2,000 genes), ADNI proteomics (n = 456 for AUC, n = 620 for Cox), ADNI transcriptomics (n = 305 for AUC, n = 518 for Cox), ADNI gene expression microarray (745 subjects), TMS brain scRNA-seq (5,000 cells).
- **Distinct vs. repeated measurements:** All measurements were taken from distinct samples (distinct subjects for ADNI analyses; distinct cells for scRNA-seq).
- **Statistical tests:** Two-sided throughout. Cox proportional hazards regression (Wald test), log-rank test for Kaplan–Meier, Spearman correlation, AUC via 5-fold stratified cross-validation (5 repeats). All described in Methods.
- **Covariates:** Cox models adjusted for APOE4 carrier status (binary).
- **Assumptions/corrections:** No multiple-comparison correction applied to discovery-phase VK analysis (exploratory); clinical validation used p < 0.05 threshold. Cox proportional hazards assumption verified.
- **Effect sizes:** Reported as hazard ratios (HR) with 95% confidence intervals, Cohen's d for GenKI knockout effects, Spearman ρ for correlations.
- **P values:** Exact values reported throughout.

## 2. Software and code

**Data collection:** No primary data collection. All datasets were obtained from public repositories (GEO GSE329430, ADNI, BioStudies S-BSST1423, figshare doi:10.6084/m9.figshare.8278114).

**Data analysis:** Custom Neural ODE + Sinkhorn OT framework (Python, PyTorch). All analysis code is available in a public GitHub repository at https://github.com/dong-zhongyuan/AD-Multi-Omics-Integration under the MIT license, accessible to reviewers at the time of submission. Key software: scanpy 1.11.5, gseapy 1.3.0, lifelines 0.27.7, R packages ggplot2/circlize/ggridges/ggsankeyfier. Full version list in requirements.txt.

**AI-use disclosure:** Large language models (Z.ai GLM and OpenAI GPT) were used as writing assistants for code drafting, language editing, and formatting of the manuscript text and figures. All scientific content, analyses, interpretations, and conclusions were conceived, verified, and approved by the authors, who take full responsibility for the integrity and accuracy of all reported results.

## 3. Data

**Data availability statement:** See manuscript Methods section "Data availability." All datasets are publicly accessible: 5xFAD scRNA-seq (GEO GSE329430), TMS brain scRNA-seq (figshare doi:10.6084/m9.figshare.8278114), tissue-specific PPI atlas (BioStudies S-BSST1423), MGI orthology database (HOM_MouseHumanSequence.rpt). ADNI data are restricted and available by application at https://adni.loni.usc.edu. Drug evidence data from OpenTargets, DGIdb, ChEMBL, PubChem (all public APIs).

## 4. Research involving human participants

**Reporting on sex and gender:** ADNI cohort includes both male and female participants. Sex was not used as a stratification variable in the analysis. APOE4 genotype (not sex) was the primary covariate. Sex-based analysis was not performed because the study is a computational re-analysis of existing data, not a prospective clinical trial.

**Reporting on race, ethnicity, or other socially relevant groupings:** Not collected or analyzed. ADNI demographic data include race/ethnicity but these were not used as variables in our analysis.

**Population characteristics:** ADNI baseline cohort: CN (n = 1,202), MCI (n = 1,292), AD (n = 462). For diagnostic analysis: CN vs. AD (n = 456 proteomics, n = 305 transcriptomics). For survival analysis: all subjects with baseline MMSE (n = 620 proteomics, n = 518 transcriptomics). Mean age ~75 years. The 5xFAD mouse discovery cohort: 10,000 cells (5,000 brain + 5,000 bone marrow) from transgenic mice and wild-type littermate controls.

**Recruitment:** Participants were recruited by the ADNI study (https://adni.loni.usc.edu) according to its published inclusion/exclusion criteria. No additional recruitment was performed for this computational study. Potential self-selection bias is inherent in the ADNI cohort (voluntary participation; predominantly North American).

**Ethics oversight:** ADNI was approved by the institutional review boards of all participating sites. All participants provided written informed consent. This computational study used only de-identified, previously collected ADNI data. No new human data were collected.

## 5. Field-specific reporting

**Selection:** Life sciences

## 6. Life sciences study design

**Sample size:** No sample size calculation was performed. All available subjects with matching data (proteomics + diagnosis, or transcriptomics + diagnosis) from the ADNI cohort were used. The 5xFAD discovery cohort used all 10,000 quality-filtered cells.

**Data exclusions:** Quality filtering applied: scRNA-seq cells with <200 genes or >20% mitochondrial reads were excluded. ADNI NULISA samples failing QC (SampleQC ≠ "passed") were excluded. No data were excluded after these pre-established QC criteria.

**Replication:** The Neural ODE training used fixed random seeds for reproducibility. Cross-validation (5-fold, 5 repeats) was used for diagnostic AUC estimation. The Jacobian AUC benchmark was computed on held-out validation data (n = 1,400 samples). No biological replication experiments were performed (computational study).

**Randomization:** Not applicable. This is a retrospective computational study using existing data; no experimental groups were randomized. For diagnostic model training, data were randomly split into training/validation (90/10) with fixed seed.

**Blinding:** Not applicable. Computational analysis of de-identified data; no assessors or operators to blind.

## 7. Animals and other research organisms

**Laboratory animals:** 5xFAD transgenic mice and wild-type littermate controls. The scRNA-seq data (GEO GSE329430) were generated by the original study; age and sex details are available in the original publication. This study performed only computational re-analysis of these publicly available data. No new animal experiments were conducted.

**Wild animals:** Not involved.

**Reporting on sex:** Both male and female mice were included in the original 5xFAD dataset. Sex was not used as a variable in our analysis.

**Field-collected samples:** Not involved.

**Ethics oversight:** The original 5xFAD scRNA-seq data were generated under approved animal care protocols by the data-generating authors. No new animal work was performed in this study.

## 8. Clinical data

**Clinical trial registration:** Not a clinical trial. This is a computational/observational study using existing ADNI data.

**Study protocol:** No new study protocol. ADNI study protocol available at https://adni.loni.usc.edu.

**Data collection:** Retrospective analysis of existing data. No new data collected.

**Outcomes:** Primary: diagnostic discrimination (AUC for AD vs. CN), prognostic prediction (Cox HR for cognitive decline). Secondary: drug evidence scoring (OpenTargets phase + tractability).

## 9. Dual use research of concern

**Hazards:** No. The work poses no threat to public health, national security, crops/livestock, or ecosystems. The computational predictions require experimental validation before clinical application.

**Experiments of concern:** No.

## 10. Materials & experimental systems

- Antibodies: Not involved
- Eukaryotic cell lines: Not involved
- Palaeontology: Not involved
- Animals and other organisms: Involved (5xFAD mouse data — computational re-analysis only)
- Clinical data: Involved (ADNI cohort — computational re-analysis only)
- Dual use research: Not involved
- Plants: Not involved

## 11. Methods

- ChIP-seq: Not involved
- Flow cytometry: Not involved
- MRI-based neuroimaging: Not involved
