# Cross-Tissue Neural ODE Framework — Replication Package

Code and instructions to reproduce:

> *A Neural Ordinary Differential Equation framework learns the continuous
> cross-tissue regulatory transformation between brain and blood and its transfer
> to Alzheimer's disease.*

Target venue: **Cell Reports Medicine**.

## Citation

If you use this code, please cite:

```bibtex
@article{dong2026neuralode,
  title={A Neural Ordinary Differential Equation framework learns the continuous
         cross-tissue regulatory transformation between brain and blood
         and its transfer to Alzheimer's disease},
  author={Dong, Zhongyuan and Meng, Xuanlin and Wang, Lianghua},
  journal={Cell Reports Medicine},
  year={2026},
  note={Z. Dong and X. Meng contributed equally}
}
```

> **Authors:** Zhongyuan Dong$^\dagger$, Xuanlin Meng$^{\dagger,*}$, Lianghua Wang$^*$
> ($^\dagger$ contributed equally; $^*$ corresponding)

---

## What this package contains

```
AD-Multi-Omics-Replication/
├── run_all.sh                 # master script: runs the whole pipeline end-to-end
├── README.md                  # this file
├── .gitignore
├── code/
│   ├── scripts/               # step0–step6 pipeline (Python)
│   │   ├── step0_preprocess/  #     raw → AnnData per omics layer
│   │   ├── step1_world_model/ #     Neural ODE training (CORE METHOD)
│   │   ├── step2_cross_tissue_causality/   # Jacobian edge extraction
│   │   ├── step3_hub_identification/       # MultiXrank + Gene Significance
│   │   ├── step4_virtual_knockout/         # GenKI + Geneformer validation
│   │   ├── step5_clinical/                 # diagnostic panel + survival
│   │   └── step6_external_validation/      # independent AD brain validation
│   ├── tools/                 # shared utilities + vendored Sinkhorn/ODE (hepaworld/)
│   └── figures/               # R + Python panel scripts for the 3 main figures
├── config/                    # project_config.yaml (paths, genes, parameters)
├── results/                   # small summary CSVs (for number verification)
├── environment/
│   └── requirements.txt       # pinned Python dependencies
└── docs/
    ├── DATA_AND_CODE.md       # data/model/software access (incl. restricted ADNI)
    └── script_to_output_map.md  # manuscript-number → script → output traceability
```

**Not included (by design):** raw data (3.4 GB; see `docs/DATA_AND_CODE.md`),
large processed `.h5ad` intermediates (3.9 GB; regenerated), and model weights
(Geneformer; download link in docs).

---

## Quick start

```bash
# 1. environment
pip install -r environment/requirements.txt

# 2. download data (public GEO + Metabolomics Workbench; ADNI by application)
#    see docs/DATA_AND_CODE.md, then point DATA_ROOT at it
export DATA_ROOT=/path/to/data

# 3. run end-to-end
bash run_all.sh
```

Wall time ≈ 8–12 h on one GPU (Step 1 transcriptomics training dominates).

---

## Reproducibility guarantees

- **Pinned versions** in `environment/requirements.txt`.
- **Fixed random seeds** (`code/tools/hepaworld/utils/seed.py`).
- **Every manuscript number** traces to a script + output file
  (`docs/script_to_output_map.md`).
- **No confidential/restricted data** committed; ADNI accessed only by application.

## License

Code: MIT. Third-party datasets retain their original licenses (see `docs/DATA_AND_CODE.md`).
