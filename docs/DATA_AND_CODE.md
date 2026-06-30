# Data and Code Availability

This document describes every input dataset, model weight, and software dependency
required to reproduce the analyses in:

> *A Neural Ordinary Differential Equation framework learns the continuous
> cross-tissue regulatory transformation between brain and blood and its transfer
> to Alzheimer's disease.*

The code is fully deposited here. **No primary data are redistributed in this
repository** — they are either public (GEO / Metabolomics Workbench) or
access-restricted (ADNI) and must be obtained under the terms below.

---

## 1. Public datasets (download via the links; no login required)

| Dataset | Role | Source | Identifier |
|---|---|---|---|
| iNPH paired CSF–PBMC scRNA-seq | Discovery cohort | GEO | **GSE292141** |
| AD brain bulk RNA-seq (BA9, entorhinal cortex) | External validation | GEO | **GSE140841** |
| AD blood/PBMC RNA-seq | Blood validation | GEO | **GSE226602** |
| Plasma metabolomics (LC-MS) | Metabolomics layer | Metabolomics Workbench | **ST000046** |
| CSF metabolomics (LC-MS) | Metabolomics layer | Metabolomics Workbench | **ST000047** |

After download, place each under `data/<dataset_name>/` matching the paths in
`config/project_config.yaml`, or set the `DATA_ROOT` environment variable.

---

## 2. Restricted data (ADNI) — access by application

The Alzheimer's Disease Neuroimaging Initiative (ADNI) provides the **proteomics**
(NULISAseq, 132 proteins, paired CSF + plasma) and the **longitudinal gene
expression + clinical outcome** data used for the diagnostic panel and survival
analysis.

- **How to obtain:** Apply at <https://adni.loni.usc.edu/data-samples/access-data/>.
  Approval is granted to qualified researchers for non-commercial research.
- **Redistribution:** **Not permitted.** ADNI data are governed by a Data Use
  Agreement and must not be copied into this repository or any public location.
- **Reproducibility for reviewers who lack ADNI access:** a pseudo/synthetic
  metabolomics + a per-feature summary of the ADNI proteomics (means, SDs, AUCs)
  is provided in `results/` so the downstream clinical figures can be reconstructed
  without the raw restricted matrices. The exact ADNI-derived summary numbers
  reported in the manuscript are in `results/` CSVs for verification.

---

## 3. Model weights & external code (download separately)

These third-party GitHub/Hugging Face repositories are imported by name from `code/tools/`
and must be cloned there before running the pipeline:

| Resource | Used by | Source (URL) | Where to place |
|---|---|---|---|
| **Geneformer V2-104M** + `token_dictionary_gc104M.pkl` | Step 4 (transformer virtual knockout) | <https://huggingface.co/ctheodoris/Geneformer> | `code/tools/geneformer-main/` |
| **GenKI** (variational graph autoencoder KO) | Step 4 (graph-autoencoder knockout) | <https://github.com/yjgeno/GenKI> (Chen et al., 2023, *Nucleic Acids Res.*) | `code/tools/GenKI-master/` |
| **MultiXrank** (multilayer random-walk centrality) | Step 3 (hub identification) | <https://github.com/anthbapt/multixrank> (Baptista et al., 2022) | `code/tools/multixrank/` |
| Sinkhorn OT + Neural ODE integrator | Step 1 (core method, **this work**) | Vendored in `code/tools/hepaworld/models/dynamics.py` (no download) | — |

Clone commands:
```bash
cd code/tools
git clone https://huggingface.co/ctheodoris/Geneformer geneformer-main
git clone https://github.com/yjgeno/GenKI GenKI-master
git clone https://github.com/anthbapt/multixrank multixrank
```

---

## 4. Software environment

- **Python:** 3.10.11  →  `pip install -r environment/requirements.txt`
- **R:** 4.4+ with `ggplot2`, `tidyverse`, `ggrepel`, `patchwork` (for figures only)
- **GPU:** a CUDA-capable GPU is required for Step 1 (Neural ODE training) and
  Step 4 (Geneformer inference). CPU-only runs are possible but slow.

Random seeds are fixed in `code/tools/hepaworld/utils/seed.py` for determinism.

---

## 5. End-to-end reproduction

```bash
pip install -r environment/requirements.txt
export DATA_ROOT=/path/to/downloaded/data
bash run_all.sh
```

Outputs land in `results/` and a figures output directory. The `results/` CSVs
shipped here let a verifier check the manuscript numbers against the regenerated
outputs (see `docs/script_to_output_map.md`).
