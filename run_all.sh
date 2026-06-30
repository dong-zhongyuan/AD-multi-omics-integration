#!/usr/bin/env bash
# =============================================================================
# Master replication script — runs the full pipeline end-to-end.
#
# Usage:
#   1. Set REPO_ROOT to the location of this replication package.
#   2. Set DATA_ROOT to where you downloaded the input datasets (see docs/DATA_AND_CODE.md).
#   3. Activate the Python environment (conda/pip install -r environment/requirements.txt).
#   4. bash run_all.sh
#
# Expected wall time: ~8-12 h on a single GPU (transcriptomics step1 dominates).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"   # input datasets (download separately)
export PYTHONPATH="$REPO_ROOT/code:${PYTHONPATH:-}"

# ── Helper ──
run() { echo; echo "▶▶ $*"; python "$@"; }

echo "============================================================"
echo "  Cross-tissue Neural ODE pipeline — master run"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  DATA_ROOT=$DATA_ROOT"
echo "============================================================"

# ── Step 0: preprocess raw inputs into per-omics AnnData ──
run "$REPO_ROOT/code/scripts/step0_preprocess/step0_preprocess_transcriptomics.py"
run "$REPO_ROOT/code/scripts/step0_preprocess/step0_preprocess_proteomics.py"
run "$REPO_ROOT/code/scripts/step0_preprocess/step0_preprocess_metabolomics.py"
run "$REPO_ROOT/code/scripts/step0_preprocess/step0_prepare_step4_vk_input_gse292141.py"

# ── Step 1: train Neural ODE world models (the core method) ──
run "$REPO_ROOT/code/scripts/step1_world_model/step1_train_transcriptomics.py"
run "$REPO_ROOT/code/scripts/step1_world_model/step1_train_proteomics.py"
run "$REPO_ROOT/code/scripts/step1_world_model/step1_train_metabolomics_common.py"
# ablation comparison (Neural ODE vs Direct OT vs Linear Ridge)
run "$REPO_ROOT/code/scripts/step1_world_model/step1_ablation_methods.py"

# ── Step 2: extract cross-tissue edges (Jacobian sensitivity) ──
run "$REPO_ROOT/code/scripts/step2_cross_tissue_causality/step2_extract_cross_tissue_edges.py"

# ── Step 3: hub identification (MultiXrank + Gene Significance) ──
run "$REPO_ROOT/code/scripts/step3_hub_identification/step3_multixrank.py"
run "$REPO_ROOT/code/scripts/step3_hub_identification/step3_analyze_eigengenes.py"

# ── Step 4: dual-method virtual knockout (GenKI + Geneformer) ──
#   NOTE: Geneformer requires the V2-104M model weights from Hugging Face.
#   See docs/DATA_AND_CODE.md §Model weights.
#   The GenKI runs use the vendored VGAE in code/tools/hepaworld/models/.
run "$REPO_ROOT/code/scripts/step4_virtual_knockout/run_geneformer_consensus_genes.py"

# ── Step 5: clinical translation (diagnostic panel + survival + drug mining) ──
#   NOTE: diagnostic & survival analyses use restricted ADNI data — see docs/DATA_AND_CODE.md §2.
run "$REPO_ROOT/code/scripts/step5_clinical/prepare_diagnostic_data.py"
run "$REPO_ROOT/code/scripts/step5_clinical/run_diagnostic_panels.py"
run "$REPO_ROOT/code/scripts/step5_clinical/step5_survival/prepare_longitudinal_survival_data.py"
run "$REPO_ROOT/code/scripts/step5_clinical/drugmining.py"

# ── Step 6: external validation in independent AD brain ──
run "$REPO_ROOT/code/scripts/step6_external_validation/run_external_validation.py"

# ── Figures (R, requires R + tidyverse + ggplot2) ──
echo; echo "▶▶ Figures (R)"; Rscript -e "sapply(list.files('code/figures/r_figures', '^fig[0-9]+\\\\.R$', full.names=TRUE), source)"

echo; echo "============================================================"
echo "  Pipeline complete. Outputs in results/ and figures output dir."
echo "============================================================"
