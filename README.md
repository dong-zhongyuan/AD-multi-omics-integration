# World Model for Brain-Blood Mapping — Replication Package

Code and instructions to reproduce:

> *World Model for Brain-Blood Mapping and Virtual Knockout Validation Uncovers
> Drug Repurposing Opportunities and Diagnostic Biomarker Candidates in
> Alzheimer's Disease*

> **Authors:** Zhongyuan Dong$^{\dagger}$, Xuanlin Meng$^{\dagger,\ddagger}$, Lianghua Wang$^*$
> ($^\dagger$ contributed equally; $^\ddagger$ co-corresponding; $^*$ corresponding)

---

## What this package contains

```
AD-Multi-Omics-Replication/
├── README.md                      # this file
├── manuscript.tex                 # full manuscript (LaTeX)
├── cover_letter.tex               # editor cover letter
├── code/
│   ├── scripts/                   # step0–step6 pipeline
│   │   ├── step0_preprocess/      #   raw → AnnData per omics layer
│   │   ├── step1_world_model/     #   Neural ODE training + benchmark (CORE)
│   │   ├── step2_cross_tissue_causality/  # Jacobian edge extraction
│   │   ├── step3_hub_identification/      # MultiXrank + hub filtering
│   │   ├── step4_virtual_knockout/        # GenKI + PPI + SCENIC + pathway enrichment
│   │   ├── step5_clinical/                # diagnostic + survival + drug mining + pseudotime
│   │   ├── step5_validation/              # VK validation analysis
│   │   └── step6_external_validation/     # independent AD brain validation
│   ├── figures/                   # all figure panel scripts (a–k per figure)
│   │   ├── Figure1/               #   world model + benchmark (11 panels)
│   │   ├── Figure2/               #   Jacobian edges + hub filtering (8 panels)
│   │   ├── Figure3/               #   tri-method VK validation (10 panels)
│   │   │   └── data_prep/         #     data preparation scripts
│   │   ├── Figure4/               #   diagnostic translation (10 panels)
│   │   └── Figure5/               #   therapeutic translation (8 panels)
│   └── tools/                     # pptx builders + KM extraction + utilities
├── config/                        # project_config.yaml + gene mappings
├── environment/
│   └── requirements.txt           # pinned Python + R dependencies
├── docs/
│   ├── DATA_AND_CODE.md           # data/model/software access (incl. restricted ADNI)
│   └── script_to_output_map.md    # manuscript-number → script → output traceability
└── tools/
    └── GenKI-master/              # vendored GenKI VGAE knockout tool
```

**Not included (by design):** raw data (3.4 GB; see `docs/DATA_AND_CODE.md`),
large processed `.h5ad` intermediates (3.9 GB; regenerated), model weights
(Geneformer; download link in docs), and ADNI clinical data (access by application only).

---

## Pipeline overview

| Step | Script directory | What it does |
|---|---|---|
| 0 | `step0_preprocess/` | Raw scRNA-seq → quality-filtered AnnData; ortholog mapping |
| 1 | `step1_world_model/` | Neural ODE + Sinkhorn OT training; Jacobian benchmark (AUC 0.893) |
| 2 | `step2_cross_tissue_causality/` | Jacobian sensitivity → cross-tissue edge network |
| 3 | `step3_hub_identification/` | MultiXrank hub ranking; edge confidence filtering |
| 4 | `step4_virtual_knockout/` | GenKI + PPI propagation + SCENIC GRN perturbation; pathway enrichment (gseapy/Enrichr) |
| 5 | `step5_clinical/` | ADNI diagnostic AUC + Cox survival + drug mining + aging pseudotime |
| 6 | `step6_external_validation/` | Independent GSE140841 brain + blood validation |

## Key results reproduced

- **Jacobian AUC** = 0.893 (Neural ODE) vs 0.560 (Ridge) → `step1_world_model/`
- **MAPT** diagnostic AUC = 0.719, Cox HR = 1.14, p = 0.003 → `step5_clinical/`
- **CSF3R** Phase II (filgrastim), **MMP9** Phase III (andecaliximab), **CXCR2** Phase III (danirixin) → `step5_clinical/step5_drugmining/`
- **Pathway enrichment**: forward → axonogenesis (p = 9.5e-14); reverse → vesicle transport (p = 2.1e-9) → `step4_virtual_knockout/run_pathway_enrichment.py`
- **Aging pseudotime**: Spearman(age, pseudotime) ρ = 0.28, p = 6e-91 → `step5_clinical/run_aging_pseudotime.py`

---

## Quick start

```bash
# 1. environment
pip install -r requirements.txt

# 2. R packages (for figure generation)
Rscript -e 'install.packages(c("ggplot2","circlize","ggridges","patchwork","rayshader")); remotes::install_github("SAngiamo/ggsankeyfier")'

# 3. download data (public GEO + figshare; ADNI by application)
#    see docs/DATA_AND_CODE.md, then point DATA_ROOT at it
export DATA_ROOT=/path/to/data

# 4. run pipeline end-to-end (step0 → step6)
cd code/scripts
python step0_preprocess/step0_preprocess_5xfad.py
python step1_world_model/step1_train_transcriptomics.py
# ... (see docs/script_to_output_map.md for full sequence)

# 5. generate figures
cd ../figures/Figure1 && python run_all.py
cd ../Figure2 && python run_all.py
cd ../Figure3 && python run_all.py  # requires R + Python
cd ../Figure4 && python run_all.py
cd ../Figure5 && python run_all.py

# 6. assemble pptx
cd ../../tools
python build_figure1_pptx.py
# ... (one per figure)
```

---

## Dependencies

### Python (environment/requirements.txt)
- Core: numpy, pandas, scipy, scikit-learn, statsmodels
- Deep learning: torch, torch-geometric, lightning
- Single-cell: scanpy, anndata, h5py
- GRN: pyscenic, arboreto
- Survival: lifelines
- Enrichment: gseapy
- Visualization: matplotlib, seaborn, plotly, python-pptx, Pillow

### R (install manually)
- ggplot2, circlize, ggridges, ggsankeyfier, patchwork, rayshader

### Vendored tools (in code/tools/)
- GenKI (variational graph autoencoder knockout) → `tools/GenKI-master/`
- Custom Neural ODE + Sinkhorn OT engine → `code/tools/hepaworld/`

---

## Reproducibility guarantees

- **Pinned versions** in `requirements.txt`.
- **Fixed random seeds** (`code/tools/hepaworld/utils/seed.py`).
- **Every manuscript number** traces to a script + output file (`docs/script_to_output_map.md`).
- **No confidential/restricted data** committed; ADNI accessed only by application.
- **Figure scripts** are self-contained: each reads a CSV/JSON from `data/` and outputs png/pdf/svg.

## License

Code: MIT. Third-party datasets and tools retain their original licenses (see `docs/DATA_AND_CODE.md`).
