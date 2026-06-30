#!/usr/bin/env Rscript
#
# 诊断靶点效力分析
# 
# 输入：Forward验证的基因（CSF → PBMC）
# 分析：ROC曲线、AUC、敏感性、特异性、最佳截断值
#

library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)

# 项目根目录
PROJECT_ROOT <- ""

# 读取诊断靶点
diagnostic_genes <- read.csv(file.path(PROJECT_ROOT, "output/step5_gene_classification/diagnostic_targets.csv"))
cat("诊断靶点:\n")
print(diagnostic_genes)

# 读取表达数据（需要根据实际数据调整）
# 这里假设有PBMC表达数据和诊断标签
# 实际使用时需要替换为真实数据路径

load_expression_data <- function() {
  # 读取PLASMA CST3诊断数据
  data_file <- file.path(PROJECT_ROOT, "data/pbmc_expression_with_diagnosis.csv")
  
  if (!file.exists(data_file)) {
    cat("\n⚠️ 数据文件不存在:", data_file, "\n")
    cat("请先运行: python3 scripts/step5_clinical/step5_diagnostic/prepare_plasma_cst3_data.py\n\n")
    return(NULL)
  }
  
  expr_data <- read.csv(data_file)
  
  cat("\n✅ 成功加载PLASMA CST3诊断数据\n")
  cat(sprintf("   样本数: %d\n", nrow(expr_data)))
  cat(sprintf("   CN: %d, AD: %d\n", sum(expr_data$diagnosis == 0), sum(expr_data$diagnosis == 1)))
  cat(sprintf("   CST3范围: %.2f - %.2f\n\n", min(expr_data$CST3), max(expr_data$CST3)))
  
  return(expr_data)
}

# ROC分析函数
perform_roc_analysis <- function(expression_data, gene_name, output_dir) {
  
  if (is.null(expression_data)) {
    cat(sprintf("跳过 %s：无表达数据\n", gene_name))
    return(NULL)
  }
  
  # 检查基因是否存在
  if (!gene_name %in% colnames(expression_data)) {
    cat(sprintf("警告：基因 %s 不在表达数据中\n", gene_name))
    return(NULL)
  }
  
  # 提取基因表达和诊断标签
  gene_expr <- expression_data[[gene_name]]
  diagnosis <- expression_data$diagnosis
  
  # 计算ROC
  roc_obj <- roc(diagnosis, gene_expr, quiet = TRUE)
  
  # 计算最佳截断值（Youden指数）
  coords_obj <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity", 
                                                  "ppv", "npv", "accuracy"))
  
  # 结果汇总
  results <- data.frame(
    gene = gene_name,
    auc = as.numeric(auc(roc_obj)),
    auc_ci_lower = as.numeric(ci.auc(roc_obj)[1]),
    auc_ci_upper = as.numeric(ci.auc(roc_obj)[3]),
    best_threshold = coords_obj$threshold,
    sensitivity = coords_obj$sensitivity,
    specificity = coords_obj$specificity,
    ppv = coords_obj$ppv,
    npv = coords_obj$npv,
    accuracy = coords_obj$accuracy,
    n_samples = length(diagnosis),
    n_ad = sum(diagnosis == 1),
    n_control = sum(diagnosis == 0)
  )
  
  # 绘制ROC曲线
  pdf(file.path(output_dir, sprintf("%s_roc_curve.pdf", gene_name)), width = 6, height = 6)
  plot(roc_obj, 
       main = sprintf("%s ROC Curve\nAUC = %.3f (95%% CI: %.3f-%.3f)", 
                      gene_name, results$auc, results$auc_ci_lower, results$auc_ci_upper),
       col = "#1f77b4", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  
  # 标记最佳截断点
  points(coords_obj$specificity, coords_obj$sensitivity, 
         pch = 19, col = "red", cex = 1.5)
  text(coords_obj$specificity - 0.1, coords_obj$sensitivity + 0.05,
       sprintf("Best cutoff\nSens: %.2f\nSpec: %.2f", 
               coords_obj$sensitivity, coords_obj$specificity),
       cex = 0.8)
  dev.off()
  
  cat(sprintf("\n%s 诊断效力:\n", gene_name))
  cat(sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)\n", 
              results$auc, results$auc_ci_lower, results$auc_ci_upper))
  cat(sprintf("  最佳截断值: %.3f\n", results$best_threshold))
  cat(sprintf("  敏感性: %.2f%%\n", results$sensitivity * 100))
  cat(sprintf("  特异性: %.2f%%\n", results$specificity * 100))
  cat(sprintf("  准确率: %.2f%%\n", results$accuracy * 100))
  cat(sprintf("  阳性预测值: %.2f%%\n", results$ppv * 100))
  cat(sprintf("  阴性预测值: %.2f%%\n", results$npv * 100))
  
  return(results)
}

# 主函数
main <- function() {
  cat(paste(rep("=", 80), collapse=""))
  cat("\n诊断靶点效力分析\n")
  cat(paste(rep("=", 80), collapse=""))
  cat("\n")
  
  # 创建输出目录
  output_dir <- file.path(PROJECT_ROOT, "output/step5_diagnostic_performance")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 加载表达数据
  expression_data <- load_expression_data()
  
  # 对每个诊断靶点进行ROC分析
  all_results <- list()
  for (i in 1:nrow(diagnostic_genes)) {
    gene <- diagnostic_genes$gene[i]
    result <- perform_roc_analysis(expression_data, gene, output_dir)
    if (!is.null(result)) {
      all_results[[gene]] <- result
    }
  }
  
  # 保存结果
  if (length(all_results) > 0) {
    results_df <- do.call(rbind, all_results)
    write.csv(results_df, 
              file.path(output_dir, "diagnostic_performance_summary.csv"),
              row.names = FALSE)
    cat(sprintf("\n结果已保存到: %s\n", output_dir))
  } else {
    cat("\n无结果生成（需要提供表达数据）\n")
    cat("\n使用说明:\n")
    cat("1. 准备PBMC表达数据，包含诊断标签\n")
    cat("2. 将数据保存为: data/pbmc_expression_with_diagnosis.csv\n")
    cat("3. 数据格式：行=样本，列=基因+diagnosis（0/1）\n")
    cat("4. 重新运行此脚本\n")
  }
}

# 运行
main()
