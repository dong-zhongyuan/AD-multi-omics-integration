# Data and Code Availability

This document describes every input dataset, model weight, and software dependency
required to reproduce the analyses in:

> *World Model for Brain-Blood Mapping and Virtual Knockout Validation Uncovers
> Drug Repurposing Opportunities and Diagnostic Biomarker Candidates in
> Alzheimer's Disease.*

The code is fully deposited here. **No primary data are redistributed in this
repository**, they are either public (GEO / figshare / Metabolomics Workbench) or
access-restricted (ADNI) and must be obtained under the terms below.

---

## 1. Public datasets (download via the links; no login required)

| Dataset | Role | Source | Identifier |
|---|---|---|---|
| 5xFAD brain + bone marrow scRNA-seq | Discovery cohort | GEO | **GSE329430** |
| AD brain bulk RNA-seq (BA9, entorhinal cortex) | External validation | GEO | **GSE140841** |
| AD blood/PBMC RNA-seq | Blood validation | GEO | **GSE226602** |
| Plasma metabolomics (LC-MS) | Metabolomics layer | Metabolomics Workbench | **ST000046** |
| CSF metabolomics (LC-MS) | Metabolomics layer | Metabolomics Workbench | **ST000047** |
| Tabula Muris Senis brain scRNA-seq | Aging pseudotime (Figure 4j) | figshare | **doi:10.6084/m9.figshare.8278114** |
| Tissue-specific PPI atlas | PPI propagation VK | BioStudies | **S-BSST1423** |
| MGI orthology database | Mouse-to-human gene mapping | MGI | **HOM_MouseHumanSequence.rpt** |

After download, place each under `data/<dataset_name>/` matching the paths in
`config/project_config.yaml`, or set the `DATA_ROOT` environment variable.

---

## 2. Restricted data (ADNI) — access by application

The Alzheimer's Disease Neuroimaging Initiative (ADNI) provides the following data
used for diagnostic, survival, and drug-mining analyses:

| ADNI dataset | Role | Identifier |
|---|---|---|
| NULISA plasma + CSF proteomics | Diagnostic AUC, Cox survival | BSHRI_PLA_CSF_NULISA |
| Gene expression microarray (49,386 probes) | Transcriptomics diagnostic + survival | ADNI_Gene_Expression_Profile |
| Longitudinal diagnosis (DXSUM) | Clinical staging, KM curves | DXSUM |
| MMSE scores | Survival event definition | MMSE |
| APOE genotyping | Cox covariate | APOERES |

- **How to obtain:** Apply at <https://adni.loni.usc.edu/data-samples/access-data/>.
  Approval is granted to qualified researchers for non-commercial research.
- **Redistribution:** **Not permitted.** ADNI data are governed by a Data Use
  Agreement and must not be copied into this repository or any public location.

---

## 3. Drug evidence databases (query via API; no download required)

| Database | Role | URL |
|---|---|---|
| OpenTargets | Drug phase, tractability, known drugs | <https://platform.opentargets.org> |
| DGIdb | Drug-gene interactions | <https://www.dgidb.org> |
| ChEMBL | Bioactivity data | <https://www.ebi.ac.uk/chembl> |
| PubChem | Bioactivity screening | <https://pubchem.ncbi.nlm.nih.gov> |
| Enrichr GO-BP 2023 | Pathway enrichment gene sets | via gseapy (`GO_Biological_Process_2023`) |

---

## 4. Model weights & external code (download separately)

These third-party GitHub/Hugging Face repositories are imported by name from `code/tools/`
and must be cloned there before running the pipeline:

| Resource | Used by | Source (URL) | Where to place |
|---|---|---|---|
| **Geneformer V2-104M** + `token_dictionary_gc104M.pkl` | Step 4 (transformer virtual knockout) | <https://huggingface.co/ctheodoris/Geneformer> | `code/tools/geneformer-main/` |
| **GenKI** (variational graph autoencoder KO) | Step 4 (graph-autoencoder knockout) | <https://github.com/yjgeno/GenKI> (Chen et al., 2023, *Nucleic Acids Res.*) | `code/tools/GenKI-master/` (vendored) |
| **MultiXrank** (multilayer random-walk centrality) | Step 3 (hub identification) | <https://github.com/anthbapt/multixrank> (Baptista et al., 2022) | `code/tools/multixrank/` |
| Sinkhorn OT + Neural ODE integrator | Step 1 (core method, **this work**) | Vendored in `code/tools/hepaworld/models/dynamics.py` | — |

Clone commands:
```bash
cd code/tools
git clone https://huggingface.co/ctheodoris/Geneformer geneformer-main
git clone https://github.com/anthbapt/multixrank multixrank
```

---

## 5. Software environment

### Python
- **Version:** 3.10.11
- **Install:** `pip install -r environment/requirements.txt`
- Key packages: torch, scanpy, pyscenic, lifelines, gseapy, plotly, python-pptx

### R
- **Version:** 4.4+
- **Install:** `Rscript -e 'install.packages(c("ggplot2","circlize","ggridges","patchwork","rayshader")); remotes::install_github("SAngiamo/ggsankeyfier")'`
- Used for: circlize chord diagrams, ggridges ridge plots, ggsankeyfier Sankey, rayshader (optional)

### GPU
- A CUDA-capable GPU is required for Step 1 (Neural ODE training) and Step 4 (Geneformer inference).
  CPU-only runs are possible but slow.

Random seeds are fixed in `code/tools/hepaworld/utils/seed.py` for determinism.

---

## 6. End-to-end reproduction

```bash
# 1. Install environment
pip install -r environment/requirements.txt
Rscript -e 'install.packages(c("ggplot2","circlize","ggridges","patchwork","rayshader")); remotes::install_github("SAngiamo/ggsankeyfier")'

# 2. Download public data + clone external tools (see sections 1 and 4 above)
export DATA_ROOT=/path/to/downloaded/data

# 3. Run pipeline (step0 → step6)
cd code/scripts
python step0_preprocess/step0_preprocess_5xfad.py
python step1_world_model/step1_train_transcriptomics.py
python step2_cross_tissue_causality/step2_extract_cross_tissue_edges.py
python step3_hub_identification/step3_multixrank.py
python step4_virtual_knockout/run_GenKI_knockout_transcriptomics.py
python step4_virtual_knockout/run_ppi_propagation_vk.py
python step4_virtual_knockout/run_scenic_vk.py
python step4_virtual_knockout/run_pathway_enrichment.py
python step5_clinical/step5_survival/step5_survival_analysis.py
python step5_clinical/step5_diagnostic/run_diagnostic_panels.py
python step5_clinical/step5_drugmining/drugmining.py
python step5_clinical/run_aging_pseudotime.py

# 4. Generate figures
cd ../figures
for fig in Figure1 Figure2 Figure3 Figure4 Figure5; do
    cd $fig && python run_all.py && cd ..
done

# 5. Assemble pptx
cd ../tools
for f in 1 2 3 4 5; do python build_figure${f}_pptx.py; done
```

---

## 7. Figure script structure

Each figure has self-contained panel scripts in `code/figures/Figure{N}/`:
- Each panel script reads a CSV/JSON from `data/` and outputs png/pdf/svg
- `run_all.py` regenerates all panels in one command
- Panel letters (a, b, c...) are added in the pptx assembly step, not in the figures
- Python panels use: `FS = round(5.6 × fig_h)`, `FSL = round(6.4 × fig_h)` for font sizing
- R panels use: `base_size` / `cex` calibrated to match

Figure 3 and Figure 5 include R scripts (circlize, ggridges, ggsankeyfier) alongside Python scripts.
