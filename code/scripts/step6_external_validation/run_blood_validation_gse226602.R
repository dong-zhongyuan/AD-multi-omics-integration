#!/usr/bin/env Rscript
# ===========================================================================
# Step 6 Blood Validation: GSE226602 (Gate Lab, Northwestern)
# ===========================================================================
# 数据集: PBMC scRNA-seq from AD patients and healthy controls
# 来源: Nature Medicine 2024 (PMID: 38340719)
# 样本: ~50 donors (25 AD + 25 HC), matched for age/sex/APOE genotype
# 平台: 10x Genomics Chromium 5' v2, Illumina NovaSeq 6000
# 格式: RDS (lognorm expression matrix, genes x cells)
#
# 验证策略:
#   1. 读取 lognorm expression matrix
#   2. 从 sample metadata 提取 AD/HC 标签
#   3. Pseudobulk: 每个 donor 取 mean expression
#   4. 计算 Blood_signature score (z-score mean)
#   5. ROC-AUC + permutation p-value 评估区分能力
# ===========================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(pROC)
})

cat("=" , rep("=", 69), "\n", sep="")
cat("[Blood] GSE226602 Validation (PBMC scRNA-seq, 50 donors)\n")
cat("=" , rep("=", 69), "\n", sep="")

# ============================================================================
# 配置
# ============================================================================
PROJECT_ROOT <- "."
DATA_DIR <- file.path(PROJECT_ROOT, "data/external-validation/GSE226602")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output/step6_external_validation")
RESULTS_DIR <- file.path(OUTPUT_DIR, "results")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# 动态加载 Blood_signature 基因列表
# ============================================================================
load_blood_signature <- function() {
  verified_file <- file.path(PROJECT_ROOT, "output/verified_cross_tissue_edges.csv")
  if (!file.exists(verified_file)) {
    stop("验证结果文件不存在: ", verified_file, "\n请先运行 step4")
  }
  verified_df <- read.csv(verified_file)
  # target 列 = 血端基因 (Jacobian 边方向: brain→blood)
  blood_genes <- sort(unique(verified_df$target))
  cat(sprintf("  Blood_signature: %d genes\n", length(blood_genes)))
  cat(sprintf("  Genes: %s\n", paste(blood_genes, collapse=", ")))
  return(blood_genes)
}

# Also load brain signature for internal control
load_brain_signature <- function() {
  eigengene_dir <- file.path(PROJECT_ROOT, "output/step3_hub_identification/eigengene_analysis")
  
  # Transcriptomics overlap
  trans_file <- file.path(eigengene_dir, "transcriptomics/brain_overlap_hub_disease.csv")
  if (!file.exists(trans_file)) {
    warning("转录组 overlap 文件不存在: ", trans_file)
    return(character(0))
  }
  trans_df <- read.csv(trans_file)
  # These are Ensembl IDs - need conversion
  # For now, use the gene_name column if available, otherwise skip
  if ("gene_name" %in% colnames(trans_df)) {
    trans_genes <- trans_df$gene_name
  } else {
    # Try to convert from Ensembl - use biomaRt or a local mapping
    # For robustness, read from the verified edges source column
    trans_genes <- character(0)
  }
  
  # Proteomics overlap
  prot_file <- file.path(eigengene_dir, "proteomics/brain_overlap_hub_disease.csv")
  if (!file.exists(prot_file)) {
    warning("蛋白组 overlap 文件不存在: ", prot_file)
    return(trans_genes)
  }
  prot_df <- read.csv(prot_file)
  prot_genes <- prot_df$gene
  # Remove BD- prefix
  prot_genes <- sub("^BD-", "", prot_genes)
  
  brain_genes <- sort(unique(c(trans_genes, prot_genes)))
  cat(sprintf("  Brain_signature (internal control): %d genes\n", length(brain_genes)))
  return(brain_genes)
}

BLOOD_SIGNATURE <- load_blood_signature()
BRAIN_SIGNATURE <- load_brain_signature()

# ============================================================================
# 读取 GSE226602 数据
# ============================================================================
cat("\n--- Loading GSE226602 data ---\n")

rds_file <- file.path(DATA_DIR, "GSE226602_rna_lognorm_expression.rds.gz")
if (!file.exists(rds_file)) {
  # Try uncompressed
  rds_file <- file.path(DATA_DIR, "GSE226602_rna_lognorm_expression.rds")
}
if (!file.exists(rds_file)) {
  stop("数据文件不存在! 请先运行 download_gse226602.sh 下载数据。\n",
       "Expected: ", file.path(DATA_DIR, "GSE226602_rna_lognorm_expression.rds.gz"))
}

cat("  Reading RDS file (this may take a few minutes)...\n")
expr_mat <- readRDS(rds_file)
cat(sprintf("  Matrix dimensions: %d genes x %d cells\n", nrow(expr_mat), ncol(expr_mat)))

# ============================================================================
# 提取样本信息 (从 cell barcodes 推断 donor ID 和 condition)
# ============================================================================
# GSE226602 cell barcodes format: typically "barcode_donorID"
# Sample names from GEO: GEX_PBMC_{ID}_GEX_{Condition}
# We need to map cells to donors. The RDS likely has donor info in colnames or metadata.

cat("\n--- Extracting sample metadata ---\n")

# Check if the matrix has metadata attributes
cell_names <- colnames(expr_mat)
cat(sprintf("  Example cell names: %s\n", paste(head(cell_names, 3), collapse=", ")))

# The Gate lab data typically encodes donor in the cell barcode suffix
# Format: ATCG...ATCG-1_DonorID or similar
# Let's check the pattern
sample_pattern <- unique(sub("^[ACGT]+-", "", sub("_.*$", "", cell_names)))
cat(sprintf("  Unique barcode suffixes (first 10): %s\n", 
            paste(head(sample_pattern, 10), collapse=", ")))

# Alternative: try to extract donor from the barcode
# The standard 10x format is BARCODE-1 where 1 is the GEM well
# In multiplexed data, the suffix after the barcode indicates the sample

# Strategy: parse cell names to get donor assignment
# Common patterns:
# 1. "BARCODE_DONOR" 
# 2. "BARCODE-DONOR"
# 3. Stored in a separate metadata column

# Let's try to detect the pattern
if (grepl("_", cell_names[1])) {
  # Pattern: BARCODE_SOMETHING
  parts <- strsplit(cell_names[1], "_")[[1]]
  cat(sprintf("  Cell name parts (split by _): %s\n", paste(parts, collapse=" | ")))
  
  # Try extracting donor ID as the last part after splitting
  donor_ids <- sapply(strsplit(cell_names, "_"), function(x) x[length(x)])
  
} else if (grepl("-", cell_names[1])) {
  # Pattern: BARCODE-N
  parts <- strsplit(cell_names[1], "-")[[1]]
  cat(sprintf("  Cell name parts (split by -): %s\n", paste(parts, collapse=" | ")))
  donor_ids <- sapply(strsplit(cell_names, "-"), function(x) x[length(x)])
}

unique_donors <- unique(donor_ids)
cat(sprintf("  Detected %d unique donors/samples\n", length(unique_donors)))
cat(sprintf("  Donor IDs (first 10): %s\n", paste(head(unique_donors, 10), collapse=", ")))

# ============================================================================
# Map donors to AD/HC condition
# ============================================================================
# From GEO sample list, we know the mapping:
# GSM7080013 = PBMC_1028 = Healthy Control
# GSM7080014 = PBMC_1241 = Alzheimers Disease
# etc.
# We need to create this mapping from the GEO metadata

# Hard-code the mapping from GEO (extracted from sample titles)
# This is the ONLY place where we use GEO metadata - the gene lists are dynamic
donor_condition_map <- c(
  "1028" = "HC", "1241" = "AD", "906" = "AD", "1034" = "AD",
  "1055" = "AD", "1092" = "AD", "1120" = "AD", "1160" = "AD",
  "254" = "AD", "1052" = "AD", "516" = "AD", "696" = "AD",
  "1273" = "HC", "738" = "AD", "911" = "HC", "1180" = "HC",
  "773" = "AD", "863" = "AD", "1279" = "HC", "598" = "HC",
  "917" = "AD", "932" = "AD", "780" = "HC", "802" = "AD",
  "1020" = "HC", "942" = "AD", "968" = "HC", "1147" = "AD",
  "781" = "HC", "1111" = "HC", "656" = "AD", "912" = "HC",
  "978" = "HC", "1282" = "HC", "953" = "AD", "820" = "HC",
  "1162" = "HC", "989" = "HC", "1010" = "HC", "965" = "AD",
  "921" = "AD", "230" = "AD", "836" = "HC", "1236" = "AD",
  "1237" = "AD", "1200" = "HC", "947" = "AD", "70" = "AD",
  "970" = "HC", "905" = "HC"
)

cat(sprintf("\n  Condition mapping: %d AD, %d HC\n",
            sum(donor_condition_map == "AD"), sum(donor_condition_map == "HC")))

# Try to match detected donor_ids to the condition map
matched <- donor_ids %in% names(donor_condition_map)
cat(sprintf("  Cells matched to known donors: %d / %d (%.1f%%)\n",
            sum(matched), length(matched), 100*mean(matched)))

# If matching fails, try alternative parsing
if (mean(matched) < 0.5) {
  cat("  Low match rate - trying alternative donor ID extraction...\n")
  
  # Try different parsing strategies
  # Strategy 2: second-to-last element
  donor_ids_v2 <- sapply(strsplit(cell_names, "_"), function(x) {
    if (length(x) >= 2) x[length(x)-1] else x[1]
  })
  matched_v2 <- donor_ids_v2 %in% names(donor_condition_map)
  
  if (mean(matched_v2) > mean(matched)) {
    donor_ids <- donor_ids_v2
    matched <- matched_v2
    cat(sprintf("  Strategy 2 improved: %d / %d matched (%.1f%%)\n",
                sum(matched), length(matched), 100*mean(matched)))
  }
  
  # Strategy 3: try all substrings
  if (mean(matched) < 0.5) {
    # Extract any numeric substring that matches a known donor
    known_ids <- names(donor_condition_map)
    donor_ids_v3 <- sapply(cell_names, function(cn) {
      nums <- regmatches(cn, gregexpr("[0-9]+", cn))[[1]]
      hit <- nums[nums %in% known_ids]
      if (length(hit) > 0) hit[1] else NA
    })
    matched_v3 <- !is.na(donor_ids_v3)
    
    if (mean(matched_v3) > mean(matched)) {
      donor_ids <- donor_ids_v3
      matched <- matched_v3
      cat(sprintf("  Strategy 3 (numeric match): %d / %d matched (%.1f%%)\n",
                  sum(matched), length(matched), 100*mean(matched)))
    }
  }
}

if (mean(matched) < 0.1) {
  cat("\n  ⚠️  Cannot reliably map cells to donors.\n")
  cat("  Attempting to use column metadata if available...\n")
  
  # Check if the RDS object has attributes with metadata
  if (!is.null(attr(expr_mat, "metadata"))) {
    cat("  Found metadata attribute!\n")
  }
  
  # Save diagnostic info and exit gracefully
  writeLines(c(
    paste("Cell name examples:", paste(head(cell_names, 20), collapse="\n")),
    paste("\nUnique patterns:", paste(head(unique(donor_ids), 30), collapse="\n"))
  ), file.path(RESULTS_DIR, "GSE226602_debug_cellnames.txt"))
  
  cat("  Saved debug info to GSE226602_debug_cellnames.txt\n")
  cat("  Please check cell name format and update parsing logic.\n")
  quit(status = 1)
}

# ============================================================================
# Pseudobulk aggregation
# ============================================================================
cat("\n--- Pseudobulk aggregation ---\n")

# Filter to matched cells only
expr_matched <- expr_mat[, matched]
donors_matched <- donor_ids[matched]
conditions_matched <- donor_condition_map[donors_matched]

unique_donors_matched <- unique(donors_matched)
cat(sprintf("  Using %d cells from %d donors\n", ncol(expr_matched), length(unique_donors_matched)))

# Pseudobulk: mean expression per donor
cat("  Computing pseudobulk (mean per donor)...\n")
pseudobulk <- do.call(cbind, lapply(unique_donors_matched, function(d) {
  cells_idx <- which(donors_matched == d)
  if (length(cells_idx) == 1) {
    expr_matched[, cells_idx, drop=FALSE]
  } else {
    Matrix::rowMeans(expr_matched[, cells_idx])
  }
}))
colnames(pseudobulk) <- unique_donors_matched

# Get conditions for pseudobulk samples
pb_conditions <- donor_condition_map[unique_donors_matched]
cat(sprintf("  Pseudobulk: %d donors (AD=%d, HC=%d)\n",
            length(pb_conditions), sum(pb_conditions=="AD"), sum(pb_conditions=="HC")))

# ============================================================================
# Signature score calculation
# ============================================================================
calculate_signature_score <- function(expr_mat, genes, label="") {
  available <- genes[genes %in% rownames(expr_mat)]
  cat(sprintf("  [%s] %d/%d genes found in expression matrix\n", 
              label, length(available), length(genes)))
  
  if (length(available) < 2) {
    cat(sprintf("  [%s] Too few genes available, skipping\n", label))
    return(NULL)
  }
  
  cat(sprintf("  [%s] Available genes: %s\n", label, paste(available, collapse=", ")))
  
  # Extract expression for signature genes
  subset_expr <- as.matrix(expr_mat[available, , drop=FALSE])
  
  # Z-score normalize each gene across samples
  z_scores <- t(scale(t(subset_expr)))
  
  # Mean z-score per sample = signature score
  scores <- colMeans(z_scores, na.rm = TRUE)
  
  return(list(scores = scores, genes_used = available))
}

# ============================================================================
# ROC analysis with bootstrap CI and permutation p-value
# ============================================================================
run_roc_analysis <- function(y_true, y_score, label="", n_boot=1000, n_perm=1000) {
  # y_true: 1=AD, 0=HC
  # y_score: signature score (higher = more AD-like)
  
  roc_obj <- roc(y_true, y_score, quiet=TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  # If AUC < 0.5, flip direction
  flipped <- FALSE
  if (auc_val < 0.5) {
    roc_obj <- roc(y_true, -y_score, quiet=TRUE)
    auc_val <- as.numeric(auc(roc_obj))
    y_score <- -y_score
    flipped <- TRUE
  }
  
  # Bootstrap 95% CI
  set.seed(42)
  boot_aucs <- replicate(n_boot, {
    idx <- sample(length(y_true), replace=TRUE)
    if (length(unique(y_true[idx])) < 2) return(NA)
    tryCatch(as.numeric(auc(roc(y_true[idx], y_score[idx], quiet=TRUE))),
             error = function(e) NA)
  })
  boot_aucs <- boot_aucs[!is.na(boot_aucs)]
  ci_lower <- quantile(boot_aucs, 0.025)
  ci_upper <- quantile(boot_aucs, 0.975)
  
  # Permutation p-value
  set.seed(42)
  perm_aucs <- replicate(n_perm, {
    perm_y <- sample(y_true)
    if (length(unique(perm_y)) < 2) return(NA)
    tryCatch(as.numeric(auc(roc(perm_y, y_score, quiet=TRUE))),
             error = function(e) NA)
  })
  perm_aucs <- perm_aucs[!is.na(perm_aucs)]
  perm_p <- (sum(perm_aucs >= auc_val) + 1) / (length(perm_aucs) + 1)
  
  cat(sprintf("  [%s] AUC=%.3f (95%%CI: %.3f-%.3f) p=%.4f %s\n",
              label, auc_val, ci_lower, ci_upper, perm_p,
              ifelse(flipped, "(flipped)", "")))
  
  return(list(
    roc = roc_obj,
    auc = auc_val,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    perm_p = perm_p,
    flipped = flipped,
    y_true = y_true,
    y_score = y_score
  ))
}

# ============================================================================
# Run validation
# ============================================================================
cat("\n--- Running ROC validation ---\n")

y_true <- as.integer(pb_conditions == "AD")

results_list <- list()

# Blood signature (primary)
blood_result <- calculate_signature_score(pseudobulk, BLOOD_SIGNATURE, "Blood_signature")
if (!is.null(blood_result)) {
  blood_roc <- run_roc_analysis(y_true, blood_result$scores, "Blood_signature")
  results_list[["Blood_signature"]] <- blood_roc
}

# Brain signature (internal control - should NOT work well in blood)
if (length(BRAIN_SIGNATURE) > 0) {
  brain_result <- calculate_signature_score(pseudobulk, BRAIN_SIGNATURE, "Brain_signature_control")
  if (!is.null(brain_result)) {
    brain_roc <- run_roc_analysis(y_true, brain_result$scores, "Brain_signature_control")
    results_list[["Brain_signature_control"]] <- brain_roc
  }
}

# Full signature
full_genes <- sort(unique(c(BLOOD_SIGNATURE, BRAIN_SIGNATURE)))
full_result <- calculate_signature_score(pseudobulk, full_genes, "Full_signature")
if (!is.null(full_result)) {
  full_roc <- run_roc_analysis(y_true, full_result$scores, "Full_signature")
  results_list[["Full_signature"]] <- full_roc
}

# ============================================================================
# Save results
# ============================================================================
cat("\n--- Saving results ---\n")

# Summary table
summary_df <- data.frame(
  signature = names(results_list),
  auc = sapply(results_list, function(x) x$auc),
  ci_lower = sapply(results_list, function(x) x$ci_lower),
  ci_upper = sapply(results_list, function(x) x$ci_upper),
  perm_p = sapply(results_list, function(x) x$perm_p),
  flipped = sapply(results_list, function(x) x$flipped),
  stringsAsFactors = FALSE
)
rownames(summary_df) <- NULL

write.csv(summary_df, file.path(RESULTS_DIR, "GSE226602_blood_validation_results.csv"),
          row.names = FALSE)
cat("  Saved: GSE226602_blood_validation_results.csv\n")

# ROC plot
pdf(file.path(FIGURE_DIR, "GSE226602_blood_ROC.pdf"), width=6, height=5)
plot(results_list[["Blood_signature"]]$roc, 
     main="Blood Signature Validation (GSE226602 PBMC)",
     col="darkred", lwd=2,
     print.auc=TRUE, print.auc.y=0.4)
if (!is.null(results_list[["Brain_signature_control"]])) {
  plot(results_list[["Brain_signature_control"]]$roc, add=TRUE,
       col="gray50", lwd=1.5, lty=2,
       print.auc=TRUE, print.auc.y=0.3)
}
legend("bottomright", 
       legend=c(
         sprintf("Blood_signature (AUC=%.3f, p=%.4f)", 
                 results_list[["Blood_signature"]]$auc,
                 results_list[["Blood_signature"]]$perm_p),
         if (!is.null(results_list[["Brain_signature_control"]])) 
           sprintf("Brain_signature control (AUC=%.3f)", 
                   results_list[["Brain_signature_control"]]$auc)
       ),
       col=c("darkred", "gray50"), lwd=c(2, 1.5), lty=c(1, 2))
dev.off()
cat("  Saved: GSE226602_blood_ROC.pdf\n")

# PNG version
png(file.path(FIGURE_DIR, "GSE226602_blood_ROC.png"), width=600, height=500, res=100)
plot(results_list[["Blood_signature"]]$roc, 
     main="Blood Signature Validation (GSE226602 PBMC)",
     col="darkred", lwd=2,
     print.auc=TRUE, print.auc.y=0.4)
if (!is.null(results_list[["Brain_signature_control"]])) {
  plot(results_list[["Brain_signature_control"]]$roc, add=TRUE,
       col="gray50", lwd=1.5, lty=2,
       print.auc=TRUE, print.auc.y=0.3)
}
legend("bottomright", 
       legend=c(
         sprintf("Blood_signature (AUC=%.3f, p=%.4f)", 
                 results_list[["Blood_signature"]]$auc,
                 results_list[["Blood_signature"]]$perm_p),
         if (!is.null(results_list[["Brain_signature_control"]])) 
           sprintf("Brain_signature control (AUC=%.3f)", 
                   results_list[["Brain_signature_control"]]$auc)
       ),
       col=c("darkred", "gray50"), lwd=c(2, 1.5), lty=c(1, 2))
dev.off()
cat("  Saved: GSE226602_blood_ROC.png\n")

# Print final summary
cat("\n")
cat("=" , rep("=", 69), "\n", sep="")
cat("FINAL RESULTS - GSE226602 Blood Validation\n")
cat("=" , rep("=", 69), "\n", sep="")
cat(sprintf("Dataset: GSE226602 (Gate Lab, Northwestern)\n"))
cat(sprintf("Samples: %d donors (%d AD, %d HC)\n",
            length(pb_conditions), sum(pb_conditions=="AD"), sum(pb_conditions=="HC")))
cat(sprintf("Platform: 10x Genomics 5' scRNA-seq, pseudobulk aggregation\n"))
cat("\n")
print(summary_df)
cat("\n")
cat("Done!\n")
