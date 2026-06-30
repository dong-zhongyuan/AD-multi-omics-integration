#!/usr/bin/env Rscript
# ============================================================================
# Step5 通用函数库
# 动态读取Step3输出的基因列表
# ============================================================================

library(dplyr)

# 项目路径
PROJECT_ROOT <- ""
STEP5_OUTPUT_DIR <- file.path(PROJECT_ROOT, "output/step5_clinical_validation")

#' 读取所有基因列表
#' @return data.frame with columns: ensembl_id, gene_symbol
get_all_genes <- function() {
  genes_file <- file.path(STEP5_OUTPUT_DIR, "all_genes.csv")
  
  if (!file.exists(genes_file)) {
    stop(paste("基因列表文件不存在:", genes_file, 
               "\n请先运行: python3 scripts/step5_clinical/utils/extract_genes_from_step3.py"))
  }
  
  genes <- read.csv(genes_file, stringsAsFactors = FALSE)
  return(genes)
}

#' 读取source基因列表
#' @return data.frame with columns: ensembl_id, gene_symbol
get_source_genes <- function() {
  genes_file <- file.path(STEP5_OUTPUT_DIR, "source_genes.csv")
  
  if (!file.exists(genes_file)) {
    stop(paste("Source基因列表文件不存在:", genes_file))
  }
  
  genes <- read.csv(genes_file, stringsAsFactors = FALSE)
  return(genes)
}

#' 读取target基因列表
#' @return data.frame with columns: ensembl_id, gene_symbol
get_target_genes <- function() {
  genes_file <- file.path(STEP5_OUTPUT_DIR, "target_genes.csv")
  
  if (!file.exists(genes_file)) {
    stop(paste("Target基因列表文件不存在:", genes_file))
  }
  
  genes <- read.csv(genes_file, stringsAsFactors = FALSE)
  return(genes)
}

#' 读取边列表（带基因symbol）
#' @return data.frame with Step3 edges plus gene symbols
get_edges_with_symbols <- function() {
  edges_file <- file.path(STEP5_OUTPUT_DIR, "edges_with_symbols.csv")
  
  if (!file.exists(edges_file)) {
    stop(paste("边列表文件不存在:", edges_file))
  }
  
  edges <- read.csv(edges_file, stringsAsFactors = FALSE)
  return(edges)
}

#' 获取基因symbol向量（用于循环分析）
#' @param type "all", "source", "target"
#' @return character vector of gene symbols
get_gene_symbols <- function(type = "all") {
  if (type == "all") {
    genes <- get_all_genes()
  } else if (type == "source") {
    genes <- get_source_genes()
  } else if (type == "target") {
    genes <- get_target_genes()
  } else {
    stop("type must be 'all', 'source', or 'target'")
  }
  
  return(genes$gene_symbol)
}

#' 打印基因列表摘要
print_gene_summary <- function() {
  all_genes <- get_all_genes()
  source_genes <- get_source_genes()
  target_genes <- get_target_genes()
  edges <- get_edges_with_symbols()
  
  cat("================================================================================\n")
  cat("Step3基因列表摘要\n")
  cat("================================================================================\n\n")
  
  cat(sprintf("总基因数: %d\n", nrow(all_genes)))
  cat(sprintf("Source基因数: %d\n", nrow(source_genes)))
  cat(sprintf("Target基因数: %d\n", nrow(target_genes)))
  cat(sprintf("边数: %d\n\n", nrow(edges)))
  
  cat("Source基因:\n")
  cat(paste("  ", paste(source_genes$gene_symbol, collapse=", "), "\n\n"))
  
  cat("Target基因:\n")
  cat(paste("  ", paste(target_genes$gene_symbol, collapse=", "), "\n\n"))
  
  cat("Top 5 边:\n")
  top_edges <- head(edges[order(-edges$final_score), ], 5)
  for (i in 1:nrow(top_edges)) {
    cat(sprintf("  %d. %s → %s (score=%.4f)\n", 
                i, 
                top_edges$source_symbol[i], 
                top_edges$target_symbol[i], 
                top_edges$final_score[i]))
  }
  cat("\n")
}

# 使用示例
if (FALSE) {
  # 在其他脚本中使用：
  source("scripts/step5_clinical/utils/step5_utils.R")
  
  # 打印摘要
  print_gene_summary()
  
  # 获取所有基因
  all_genes <- get_gene_symbols("all")
  
  # 获取source基因
  source_genes <- get_gene_symbols("source")
  
  # 获取target基因
  target_genes <- get_gene_symbols("target")
  
  # 获取边列表
  edges <- get_edges_with_symbols()
  
  # 循环分析每个基因
  for (gene in all_genes) {
    cat(sprintf("分析基因: %s\n", gene))
    # ... 你的分析代码 ...
  }
}
