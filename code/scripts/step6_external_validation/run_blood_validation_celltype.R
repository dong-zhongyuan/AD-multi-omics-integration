#!/usr/bin/env Rscript
# ===========================================================================
# Step 6 Blood Validation: Cell-type-specific analysis (GSE226602)
# ===========================================================================
# 策略:
#   1. 读取 lognorm expression matrix
#   2. 用经典 PBMC marker genes 注释细胞类型
#   3. 按细胞类型分别做 pseudobulk
#   4. 对每个细胞类型分别计算 signature score + ROC
#   5. 找出哪个细胞类型中 blood signature 最有区分能力
# ===========================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(pROC)
})

cat("=" , rep("=", 69), "\n", sep="")
cat("[Blood] Cell-type-specific Validation (GSE226602)\n")
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
# 加载 Blood_signature
# ============================================================================
load_blood_signature <- function() {
  verified_file <- file.path(PROJECT_ROOT, "output/verified_cross_tissue_edges.csv")
  if (!file.exists(verified_file)) {
    stop("验证结果文件不存在: ", verified_file)
  }
  verified_df <- read.csv(verified_file)
  blood_genes <- sort(unique(verified_df$target))
  cat(sprintf("  Blood_signature: %d genes\n", length(blood_genes)))
  cat(sprintf("  Genes: %s\n", paste(blood_genes, collapse=", ")))
  return(blood_genes)
}

BLOOD_SIGNATURE <- load_blood_signature()

# ============================================================================
# Donor-condition mapping (from GEO)
# ============================================================================
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

# ============================================================================
# PBMC cell type markers (经典 marker genes)
# ============================================================================
# 使用高特异性 markers 进行简单注释
cell_markers <- list(
  "CD14_Monocytes" = c("CD14", "LYZ", "S100A8", "S100A9", "VCAN"),
  "CD16_Monocytes" = c("FCGR3A", "MS4A7", "LST1", "CDKN1C"),
  "cDC" = c("FCER1A", "CD1C", "CLEC10A"),
  "pDC" = c("LILRA4", "IRF7", "CLEC4C", "IL3RA"),
  "CD4_T" = c("CD3D", "CD3E", "IL7R", "CD4", "LEF1"),
  "CD8_T" = c("CD3D", "CD3E", "CD8A", "CD8B", "GZMK"),
  "NK" = c("GNLY", "NKG7", "KLRD1", "NCAM1", "KLRF1"),
  "B_cells" = c("CD79A", "MS4A1", "CD19", "BANK1"),
  "Plasma" = c("MZB1", "JCHAIN", "XBP1", "IGHG1")
)

# ============================================================================
# 读取数据
# ============================================================================
cat("\n--- Loading GSE226602 data ---\n")

rds_file <- file.path(DATA_DIR, "GSE226602_rna_lognorm_expression.rds")
if (!file.exists(rds_file)) {
  stop("数据文件不存在: ", rds_file)
}

cat("  Reading RDS file...\n")
expr_mat <- readRDS(rds_file)
cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(expr_mat), ncol(expr_mat)))

# ============================================================================
# 提取 donor ID
# ============================================================================
cat("\n--- Extracting donor IDs ---\n")

cell_names <- colnames(expr_mat)
known_ids <- names(donor_condition_map)

# Strategy: extract numeric substring matching known donor IDs
# Cell name format: G1010_y2_AAACCTGAGCTGATAA-1
# The number after G is the donor ID (e.g., G1010 → donor 1010)
donor_ids <- sub("^G", "", sapply(strsplit(cell_names, "_"), `[`, 1))

matched <- donor_ids %in% known_ids
cat(sprintf("  Matched: %d / %d cells (%.1f%%)\n",
            sum(matched), length(matched), 100*mean(matched)))

if (mean(matched) < 0.5) {
  # Fallback: try numeric extraction
  donor_ids <- sapply(cell_names, function(cn) {
    nums <- regmatches(cn, gregexpr("[0-9]+", cn))[[1]]
    hit <- nums[nums %in% known_ids]
    if (length(hit) > 0) hit[1] else NA
  })
  matched <- !is.na(donor_ids)
  cat(sprintf("  Fallback matched: %d / %d cells (%.1f%%)\n",
              sum(matched), length(matched), 100*mean(matched)))
}

# Filter to matched cells
expr_matched <- expr_mat[, matched]
donors_matched <- donor_ids[matched]
conditions_matched <- donor_condition_map[donors_matched]

cat(sprintf("  Using %d cells from %d donors\n",
            ncol(expr_matched), length(unique(donors_matched))))

# ============================================================================
# Cell type annotation using marker gene scoring
# ============================================================================
cat("\n--- Cell type annotation ---\n")

# For each cell, calculate mean expression of each marker set
# Assign cell to the type with highest score
annotate_cells <- function(expr, markers) {
  gene_names <- rownames(expr)
  n_cells <- ncol(expr)
  
  # Calculate scores for each cell type
  scores <- matrix(0, nrow = n_cells, ncol = length(markers))
  colnames(scores) <- names(markers)
  
  for (i in seq_along(markers)) {
    ct <- names(markers)[i]
    genes <- markers[[i]]
    available <- genes[genes %in% gene_names]
    
    if (length(available) > 0) {
      # Mean expression of marker genes per cell
      if (length(available) == 1) {
        scores[, i] <- as.numeric(expr[available, ])
      } else {
        scores[, i] <- Matrix::colMeans(expr[available, ])
      }
    }
    cat(sprintf("  %s: %d/%d markers found\n", ct, length(available), length(genes)))
  }
  
  # Assign each cell to the type with max score
  cell_types <- colnames(scores)[apply(scores, 1, which.max)]
  
  # Mark cells with very low scores as "Unknown"
  max_scores <- apply(scores, 1, max)
  cell_types[max_scores < 0.1] <- "Unknown"
  
  return(cell_types)
}

cat("  Annotating cells with marker genes...\n")
cell_types <- annotate_cells(expr_matched, cell_markers)

# Summary
ct_table <- table(cell_types)
cat("\n  Cell type distribution:\n")
for (ct in sort(names(ct_table))) {
  cat(sprintf("    %s: %d cells (%.1f%%)\n", ct, ct_table[ct], 
              100 * ct_table[ct] / sum(ct_table)))
}

# ============================================================================
# Cell-type-specific pseudobulk + ROC
# ============================================================================
cat("\n--- Cell-type-specific validation ---\n")

calculate_signature_score <- function(expr_mat, genes) {
  available <- genes[genes %in% rownames(expr_mat)]
  if (length(available) < 2) return(NULL)
  
  subset <- as.matrix(expr_mat[available, , drop=FALSE])
  
  # Z-score each gene across samples, then mean
  z_scores <- t(scale(t(subset)))
  z_scores[is.na(z_scores)] <- 0
  scores <- colMeans(z_scores)
  
  return(list(scores = scores, genes_used = available))
}

run_roc_analysis <- function(y_true, scores, label) {
  # If all same class, skip
  if (length(unique(y_true)) < 2) return(NULL)
  
  roc_obj <- roc(y_true, scores, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  # If AUC < 0.5, flip
  flipped <- FALSE
  if (auc_val < 0.5) {
    roc_obj <- roc(y_true, -scores, quiet = TRUE)
    auc_val <- as.numeric(auc(roc_obj))
    flipped <- TRUE
  }
  
  # Bootstrap CI
  ci_result <- tryCatch(
    ci.auc(roc_obj, method = "bootstrap", boot.n = 1000, quiet = TRUE),
    error = function(e) NULL
  )
  ci_lower <- if (!is.null(ci_result)) as.numeric(ci_result[1]) else auc_val
  ci_upper <- if (!is.null(ci_result)) as.numeric(ci_result[3]) else auc_val
  
  # Permutation p-value
  n_perm <- 1000
  set.seed(42)
  perm_aucs <- replicate(n_perm, {
    perm_y <- sample(y_true)
    tryCatch({
      as.numeric(auc(roc(perm_y, scores, quiet = TRUE)))
    }, error = function(e) 0.5)
  })
  # Adjust for flipped
  perm_aucs <- pmax(perm_aucs, 1 - perm_aucs)
  perm_p <- (sum(perm_aucs >= auc_val) + 1) / (n_perm + 1)
  
  return(list(
    roc = roc_obj, auc = auc_val,
    ci_lower = ci_lower, ci_upper = ci_upper,
    perm_p = perm_p, flipped = flipped
  ))
}

# Run for each cell type
results_list <- list()
unique_donors <- unique(donors_matched)

for (ct in sort(unique(cell_types))) {
  if (ct == "Unknown") next
  
  # Get cells of this type
  ct_cells <- which(cell_types == ct)
  ct_donors <- donors_matched[ct_cells]
  ct_expr <- expr_matched[, ct_cells]
  
  # Need at least 5 cells per donor on average
  donor_cell_counts <- table(ct_donors)
  valid_donors <- names(donor_cell_counts[donor_cell_counts >= 10])
  
  # Need donors from both conditions
  valid_conditions <- donor_condition_map[valid_donors]
  n_ad <- sum(valid_conditions == "AD", na.rm = TRUE)
  n_hc <- sum(valid_conditions == "HC", na.rm = TRUE)
  
  if (n_ad < 5 || n_hc < 5) {
    cat(sprintf("  [%s] Skipped: too few donors (AD=%d, HC=%d)\n", ct, n_ad, n_hc))
    next
  }
  
  # Pseudobulk per donor (only valid donors)
  pb <- do.call(cbind, lapply(valid_donors, function(d) {
    cells_idx <- which(ct_donors == d)
    if (length(cells_idx) == 1) {
      ct_expr[, cells_idx, drop=FALSE]
    } else {
      Matrix::rowMeans(ct_expr[, cells_idx])
    }
  }))
  colnames(pb) <- valid_donors
  
  # Conditions
  pb_conditions <- donor_condition_map[valid_donors]
  y_true <- as.integer(pb_conditions == "AD")
  
  # Signature score
  sig_result <- calculate_signature_score(pb, BLOOD_SIGNATURE)
  if (is.null(sig_result)) {
    cat(sprintf("  [%s] Skipped: too few signature genes\n", ct))
    next
  }
  
  # ROC
  roc_result <- run_roc_analysis(y_true, sig_result$scores, ct)
  if (is.null(roc_result)) next
  
  sig_mark <- ifelse(roc_result$perm_p < 0.001, "***",
               ifelse(roc_result$perm_p < 0.01, "**",
               ifelse(roc_result$perm_p < 0.05, "*", "ns")))
  
  cat(sprintf("  [%s] AUC=%.3f (95%%CI: %.3f-%.3f) p=%.4f %s | %d donors (AD=%d, HC=%d) | %d/%d genes\n",
              ct, roc_result$auc, roc_result$ci_lower, roc_result$ci_upper,
              roc_result$perm_p, sig_mark, length(valid_donors), n_ad, n_hc,
              length(sig_result$genes_used), length(BLOOD_SIGNATURE)))
  
  results_list[[ct]] <- list(
    cell_type = ct,
    auc = roc_result$auc,
    ci_lower = roc_result$ci_lower,
    ci_upper = roc_result$ci_upper,
    perm_p = roc_result$perm_p,
    flipped = roc_result$flipped,
    n_donors = length(valid_donors),
    n_ad = n_ad,
    n_hc = n_hc,
    n_genes = length(sig_result$genes_used),
    n_cells = length(ct_cells),
    roc = roc_result$roc
  )
}

# ============================================================================
# Save results
# ============================================================================
cat("\n--- Saving results ---\n")

if (length(results_list) == 0) {
  cat("  ⚠️  No valid results\n")
  quit(status = 1)
}

# Summary table
summary_df <- data.frame(
  cell_type = sapply(results_list, `[[`, "cell_type"),
  auc = sapply(results_list, `[[`, "auc"),
  ci_lower = sapply(results_list, `[[`, "ci_lower"),
  ci_upper = sapply(results_list, `[[`, "ci_upper"),
  perm_p = sapply(results_list, `[[`, "perm_p"),
  flipped = sapply(results_list, `[[`, "flipped"),
  n_donors = sapply(results_list, `[[`, "n_donors"),
  n_ad = sapply(results_list, `[[`, "n_ad"),
  n_hc = sapply(results_list, `[[`, "n_hc"),
  n_genes = sapply(results_list, `[[`, "n_genes"),
  n_cells = sapply(results_list, `[[`, "n_cells"),
  stringsAsFactors = FALSE
)
rownames(summary_df) <- NULL

# Sort by AUC descending
summary_df <- summary_df[order(-summary_df$auc), ]

write.csv(summary_df, file.path(RESULTS_DIR, "GSE226602_celltype_validation_results.csv"),
          row.names = FALSE)
cat("  Saved: GSE226602_celltype_validation_results.csv\n")

# ROC plot - all cell types
n_ct <- length(results_list)
colors <- rainbow(n_ct)

pdf(file.path(FIGURE_DIR, "GSE226602_celltype_ROC.pdf"), width=8, height=7)
plot(0, 0, type="n", xlim=c(1, 0), ylim=c(0, 1),
     xlab="Specificity", ylab="Sensitivity",
     main="Blood Signature Validation by Cell Type (GSE226602)")
abline(a=1, b=-1, lty=2, col="gray")

legend_labels <- character(0)
sorted_results <- results_list[order(-sapply(results_list, `[[`, "auc"))]

for (i in seq_along(sorted_results)) {
  res <- sorted_results[[i]]
  plot(res$roc, add=TRUE, col=colors[i], lwd=2)
  sig_mark <- ifelse(res$perm_p < 0.05, "*", "")
  legend_labels[i] <- sprintf("%s (AUC=%.3f, p=%.3f)%s",
                               res$cell_type, res$auc, res$perm_p, sig_mark)
}

legend("bottomright", legend=legend_labels, col=colors[1:n_ct],
       lwd=2, cex=0.7, bg="white")
dev.off()
cat("  Saved: GSE226602_celltype_ROC.pdf\n")

# PNG version
png(file.path(FIGURE_DIR, "GSE226602_celltype_ROC.png"), width=800, height=700, res=100)
plot(0, 0, type="n", xlim=c(1, 0), ylim=c(0, 1),
     xlab="Specificity", ylab="Sensitivity",
     main="Blood Signature Validation by Cell Type (GSE226602)")
abline(a=1, b=-1, lty=2, col="gray")

for (i in seq_along(sorted_results)) {
  res <- sorted_results[[i]]
  plot(res$roc, add=TRUE, col=colors[i], lwd=2)
}

legend("bottomright", legend=legend_labels, col=colors[1:n_ct],
       lwd=2, cex=0.7, bg="white")
dev.off()
cat("  Saved: GSE226602_celltype_ROC.png\n")

# ============================================================================
# Final summary
# ============================================================================
cat("\n")
cat("=" , rep("=", 69), "\n", sep="")
cat("FINAL RESULTS - Cell-type-specific Blood Validation\n")
cat("=" , rep("=", 69), "\n", sep="")
cat("\n")
print(summary_df[, c("cell_type", "auc", "ci_lower", "ci_upper", "perm_p", "n_donors", "n_genes", "n_cells")])
cat("\n")

# Highlight best
best <- summary_df[1, ]
cat(sprintf("Best cell type: %s (AUC=%.3f, p=%.4f)\n", best$cell_type, best$auc, best$perm_p))

n_sig <- sum(summary_df$perm_p < 0.05)
cat(sprintf("Significant (p<0.05): %d/%d cell types\n", n_sig, nrow(summary_df)))
cat("\nDone!\n")
