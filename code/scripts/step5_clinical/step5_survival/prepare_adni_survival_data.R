#!/usr/bin/env Rscript
# ============================================================================
# ADNI生存分析数据准备
# ============================================================================
# 目标：
# 1. 提取候选基因（动态读取）的表达数据
# 2. 合并临床数据（REGISTRY, NPSTATUS, 认知评分）
# 3. 构建生存分析数据集（时间、事件、协变量）
# ============================================================================

library(tidyverse)

# 加载项目配置
library(yaml)
config <- yaml::read_yaml("config/project_config.yaml")
PROJECT_ROOT <- config$paths$project_root
DATA_DIR <- file.path(PROJECT_ROOT, "data")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")
PROCESSED_DATA_DIR <- file.path(PROJECT_ROOT, "processed-data")

library(data.table)

# 设置路径
data_dir <- file.path(DATA_DIR, "survival")
output_dir <- file.path(OUTPUT_DIR, "step5_clinical_validation/survival_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== ADNI生存分析数据准备 ===\n\n")

# ============================================================================
# 1. 读取基因表达数据
# ============================================================================
cat("1. 读取基因表达数据...\n")

# 读取完整数据（这个文件很大，需要一些时间）
expr_file <- file.path(data_dir, "ADNI_Gene_Expression_Profile.csv")
cat("   读取文件:", expr_file, "\n")
cat("   文件大小:", file.size(expr_file) / 1024^2, "MB\n")

# 使用data.table快速读取
expr_data <- fread(expr_file, header = FALSE)
cat("   数据维度:", nrow(expr_data), "行 ×", ncol(expr_data), "列\n")

# 提取元数据行
metadata_rows <- expr_data[1:8, ]
colnames(metadata_rows) <- paste0("V", 1:ncol(metadata_rows))

# 提取样本信息（第3行是SubjectID）
sample_ids <- as.character(metadata_rows[3, -c(1:3)])
visit_codes <- as.character(metadata_rows[2, -c(1:3)])
phases <- as.character(metadata_rows[1, -c(1:3)])

cat("   样本数:", length(sample_ids), "\n")
cat("   前5个样本:", paste(head(sample_ids, 5), collapse=", "), "\n")

# 提取基因注释（第9行）
gene_annotation <- expr_data[9, ]
probe_ids <- as.character(gene_annotation[[1]])
locus_links <- as.character(gene_annotation[[2]])
gene_symbols <- as.character(gene_annotation[[3]])

# 提取表达数据（第10行开始）
expr_matrix <- expr_data[10:nrow(expr_data), ]
expr_matrix <- as.data.frame(expr_matrix)
colnames(expr_matrix) <- c("ProbeSet", "LocusLink", "Symbol", sample_ids)

cat("   基因/探针数:", nrow(expr_matrix), "\n")

# ============================================================================
# 2. 提取候选基因表达数据
# ============================================================================
cat("\n2. 提取候选基因表达数据...\n")

# 动态读取治疗靶点基因列表
therapeutic_genes_file <- file.path(PROJECT_ROOT, "output/step5_gene_classification/therapeutic_genes.txt")
if (file.exists(therapeutic_genes_file)) {
  gene_symbols <- readLines(therapeutic_genes_file)
  gene_symbols <- gene_symbols[nchar(gene_symbols) > 0]
  cat("   从", therapeutic_genes_file, "读取", length(gene_symbols), "个基因\n")
} else {
  stop("治疗靶点文件不存在，请先运行 classify_validated_genes.py")
}

# 基因别名映射（CAVIN2旧名SDPR）
gene_alias <- c(CAVIN2 = "SDPR")

# 动态查找每个基因的探针
candidate_genes <- list()
for (gene in gene_symbols) {
  # 先用原名查找
  probes <- expr_matrix$ProbeSet[expr_matrix$Symbol == gene]
  # 如果没找到，尝试别名
  if (length(probes) == 0 && gene %in% names(gene_alias)) {
    alias <- gene_alias[gene]
    probes <- expr_matrix$ProbeSet[expr_matrix$Symbol == alias]
    if (length(probes) > 0) cat("   ", gene, "(alias=", alias, "):", length(probes), "个探针\n")
  }
  if (length(probes) > 0) {
    candidate_genes[[gene]] <- probes
    if (!(gene %in% names(gene_alias))) cat("   ", gene, ":", length(probes), "个探针\n")
  } else {
    cat("   ✗", gene, ": 未找到探针\n")
  }
}
cat("   成功映射:", length(candidate_genes), "/", length(gene_symbols), "个基因\n")

# 提取每个基因的表达数据
gene_expr_list <- list()

for (gene in names(candidate_genes)) {
  probes <- candidate_genes[[gene]]
  cat("   ", gene, ":", length(probes), "个探针\n")
  
  # 提取探针数据
  gene_data <- expr_matrix[expr_matrix$ProbeSet %in% probes, ]
  
  if (nrow(gene_data) == 0) {
    cat("      警告: 未找到探针数据\n")
    next
  }
  
  # 转换为数值矩阵（跳过前3列注释）
  expr_values <- gene_data[, -c(1:3)]
  expr_values <- apply(expr_values, 2, as.numeric)
  
  # 如果有多个探针，取平均值
  if (nrow(gene_data) > 1) {
    gene_expr <- colMeans(expr_values, na.rm = TRUE)
    cat("      使用", nrow(gene_data), "个探针的平均值\n")
  } else {
    # 单个探针的情况
    if (is.matrix(expr_values)) {
      gene_expr <- as.numeric(expr_values[1, ])
    } else {
      gene_expr <- as.numeric(expr_values)
    }
    cat("      使用单个探针\n")
  }
  
  gene_expr_list[[gene]] <- gene_expr
}

# 动态构建基因表达数据框
gene_expr_df <- data.frame(
  SubjectID = sample_ids,
  Visit = visit_codes,
  Phase = phases,
  stringsAsFactors = FALSE
)
for (gene in names(gene_expr_list)) {
  gene_expr_df[[gene]] <- gene_expr_list[[gene]]
}

cat("   基因表达数据框:", nrow(gene_expr_df), "行 ×", ncol(gene_expr_df), "列\n")
cat("   前5行:\n")
print(head(gene_expr_df, 5))

# 保存基因表达数据
write.csv(gene_expr_df, file.path(output_dir, "adni_gene_expression.csv"), row.names = FALSE)
cat("   已保存:", file.path(output_dir, "adni_gene_expression.csv"), "\n")

# ============================================================================
# 3. 读取临床数据
# ============================================================================
cat("\n3. 读取临床数据...\n")

# 读取REGISTRY（患者登记信息）
registry <- fread(file.path(data_dir, "REGISTRY_05May2026.csv"))
cat("   REGISTRY:", nrow(registry), "行 ×", ncol(registry), "列\n")
cat("   列名:", paste(head(colnames(registry), 10), collapse=", "), "\n")

# 读取NPSTATUS（神经病理状态）
npstatus <- fread(file.path(data_dir, "NPSTATUS_05May2026.csv"))
cat("   NPSTATUS:", nrow(npstatus), "行 ×", ncol(npstatus), "列\n")
cat("   列名:", paste(head(colnames(npstatus), 10), collapse=", "), "\n")

# 读取MMSE（认知评分）
mmse <- fread(file.path(data_dir, "MMSE_05May2026.csv"))
cat("   MMSE:", nrow(mmse), "行 ×", ncol(mmse), "列\n")

# 读取ADAS（认知评分）
adas <- fread(file.path(data_dir, "ADAS_05May2026.csv"))
cat("   ADAS:", nrow(adas), "行 ×", ncol(adas), "列\n")

# ============================================================================
# 4. 构建生存分析数据集
# ============================================================================
cat("\n4. 构建生存分析数据集...\n")

# 从SubjectID中提取RID（去掉站点前缀）
gene_expr_df$RID <- as.integer(sub(".*_S_(\\d+)", "\\1", gene_expr_df$SubjectID))
cat("   提取RID:", length(unique(gene_expr_df$RID)), "个唯一受试者\n")

# 合并REGISTRY数据（获取基线信息）
# 首先找到每个受试者的基线访视
registry_baseline <- registry %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup() %>%
  select(RID, PTID, PHASE, VISCODE, EXAMDATE, RGSTATUS)

cat("   基线访视:", nrow(registry_baseline), "个受试者\n")

# 合并基因表达数据和基线信息
survival_data <- gene_expr_df %>%
  left_join(registry_baseline, by = "RID")

cat("   合并后:", nrow(survival_data), "行\n")

# 合并NPSTATUS（神经病理状态）
# 这个数据集包含生存状态信息
npstatus_summary <- npstatus %>%
  group_by(RID) %>%
  summarise(
    NPDECIDE = ifelse(length(na.omit(NPDECIDE)) > 0, first(na.omit(NPDECIDE)), NA),
    NPDEC = ifelse(length(na.omit(NPDEC)) > 0, first(na.omit(NPDEC)), NA),
    .groups = "drop"
  )

survival_data <- survival_data %>%
  left_join(npstatus_summary, by = "RID")

cat("   合并NPSTATUS后:", nrow(survival_data), "行\n")

# 合并MMSE基线评分
mmse_baseline <- mmse %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  group_by(RID) %>%
  summarise(
    MMSE_baseline = if (length(na.omit(MMSCORE)) > 0) first(na.omit(MMSCORE)) else NA_real_,
    .groups = "drop"
  )

survival_data <- survival_data %>%
  left_join(mmse_baseline, by = "RID")

# 合并ADAS基线评分
adas_baseline <- adas %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  group_by(RID) %>%
  summarise(
    ADAS_baseline = if (length(na.omit(TOTAL13)) > 0) first(na.omit(TOTAL13)) else NA_real_,
    .groups = "drop"
  )

survival_data <- survival_data %>%
  left_join(adas_baseline, by = "RID")

cat("   合并认知评分后:", nrow(survival_data), "行\n")

# ============================================================================
# 5. 定义生存时间和事件
# ============================================================================
cat("\n5. 定义生存时间和事件...\n")

# 计算每个受试者的随访时间
# 从REGISTRY中获取每个受试者的所有访视日期
registry_followup <- registry %>%
  filter(!is.na(EXAMDATE)) %>%
  group_by(RID) %>%
  summarise(
    baseline_date = min(EXAMDATE, na.rm = TRUE),
    last_visit_date = max(EXAMDATE, na.rm = TRUE),
    n_visits = n(),
    .groups = "drop"
  ) %>%
  mutate(
    followup_days = as.numeric(difftime(last_visit_date, baseline_date, units = "days")),
    followup_years = followup_days / 365.25
  )

cat("   随访时间统计:\n")
cat("      中位随访时间:", median(registry_followup$followup_years, na.rm = TRUE), "年\n")
cat("      最长随访时间:", max(registry_followup$followup_years, na.rm = TRUE), "年\n")

survival_data <- survival_data %>%
  left_join(registry_followup, by = "RID")

# 定义事件：认知下降或转化为痴呆
# 从MMSE和ADAS的变化来定义事件
mmse_change <- mmse %>%
  group_by(RID) %>%
  filter(length(na.omit(MMSCORE)) >= 2) %>%
  summarise(
    MMSE_first = first(na.omit(MMSCORE)),
    MMSE_last = last(na.omit(MMSCORE)),
    MMSE_change = last(na.omit(MMSCORE)) - first(na.omit(MMSCORE)),
    .groups = "drop"
  )

adas_change <- adas %>%
  group_by(RID) %>%
  filter(length(na.omit(TOTAL13)) >= 2) %>%
  summarise(
    ADAS_first = first(na.omit(TOTAL13)),
    ADAS_last = last(na.omit(TOTAL13)),
    ADAS_change = last(na.omit(TOTAL13)) - first(na.omit(TOTAL13)),
    .groups = "drop"
  )

survival_data <- survival_data %>%
  left_join(mmse_change, by = "RID") %>%
  left_join(adas_change, by = "RID")

# 定义事件：
# 1. MMSE下降≥3分（临床显著下降）
# 2. ADAS增加≥4分（临床显著恶化）
survival_data <- survival_data %>%
  mutate(
    event_mmse = ifelse(!is.na(MMSE_change) & MMSE_change <= -3, 1, 0),
    event_adas = ifelse(!is.na(ADAS_change) & ADAS_change >= 4, 1, 0),
    event = ifelse(event_mmse == 1 | event_adas == 1, 1, 0)
  )

cat("   事件定义:\n")
cat("      MMSE下降≥3分:", sum(survival_data$event_mmse, na.rm = TRUE), "例\n")
cat("      ADAS增加≥4分:", sum(survival_data$event_adas, na.rm = TRUE), "例\n")
cat("      总事件数:", sum(survival_data$event, na.rm = TRUE), "例\n")

# ============================================================================
# 6. 数据清理和最终数据集
# ============================================================================
cat("\n6. 数据清理和最终数据集...\n")

# 只保留有完整数据的样本（动态基因列表）
gene_names <- names(candidate_genes)

# 动态过滤：所有基因非 NA
filter_expr <- paste0("!is.na(", gene_names, ")", collapse = " & ")
survival_final <- survival_data %>%
  filter(
    eval(parse(text = filter_expr)),
    !is.na(followup_years),
    followup_years > 0
  ) %>%
  select(
    RID, SubjectID, Visit, Phase,
    all_of(gene_names),
    MMSE_baseline, ADAS_baseline,
    followup_years, event,
    MMSE_change, ADAS_change,
    event_mmse, event_adas,
    n_visits
  )

cat("   最终数据集:", nrow(survival_final), "个样本\n")
cat("   事件数:", sum(survival_final$event), "例\n")
cat("   事件率:", round(mean(survival_final$event) * 100, 1), "%\n")

# 保存最终数据集
write.csv(survival_final, file.path(output_dir, "adni_survival_data.csv"), row.names = FALSE)
cat("   已保存:", file.path(output_dir, "adni_survival_data.csv"), "\n")

# ============================================================================
# 7. 描述性统计
# ============================================================================
cat("\n7. 描述性统计...\n")

# 基因表达统计
gene_stats <- survival_final %>%
  summarise(
    across(all_of(gene_names),
           list(mean = ~mean(., na.rm = TRUE),
                sd = ~sd(., na.rm = TRUE),
                median = ~median(., na.rm = TRUE),
                min = ~min(., na.rm = TRUE),
                max = ~max(., na.rm = TRUE)),
           .names = "{.col}_{.fn}")
  )

cat("\n基因表达统计:\n")
print(t(gene_stats))

# 临床特征统计
clinical_stats <- survival_final %>%
  summarise(
    n = n(),
    followup_mean = mean(followup_years, na.rm = TRUE),
    followup_sd = sd(followup_years, na.rm = TRUE),
    followup_median = median(followup_years, na.rm = TRUE),
    mmse_mean = mean(MMSE_baseline, na.rm = TRUE),
    mmse_sd = sd(MMSE_baseline, na.rm = TRUE),
    adas_mean = mean(ADAS_baseline, na.rm = TRUE),
    adas_sd = sd(ADAS_baseline, na.rm = TRUE),
    event_n = sum(event),
    event_rate = mean(event) * 100
  )

cat("\n临床特征统计:\n")
print(clinical_stats)

# 保存统计结果
write.csv(gene_stats, file.path(output_dir, "gene_expression_stats.csv"), row.names = FALSE)
write.csv(clinical_stats, file.path(output_dir, "clinical_stats.csv"), row.names = FALSE)

# ============================================================================
# 8. Cox回归生存分析
# ============================================================================
cat("\n8. Cox回归生存分析...\n")

library(survival)

# 对每个基因进行单变量Cox回归
cox_results <- data.frame()

for (gene in names(candidate_genes)) {
  # 单变量Cox回归
  formula_str <- paste0("Surv(followup_years, event) ~ ", gene)
  cox_model <- coxph(as.formula(formula_str), data = survival_final)
  
  # 提取结果
  summary_cox <- summary(cox_model)
  
  cox_results <- rbind(cox_results, data.frame(
    gene = gene,
    HR = summary_cox$conf.int[1, 1],
    HR_lower = summary_cox$conf.int[1, 3],
    HR_upper = summary_cox$conf.int[1, 4],
    p_value = summary_cox$coefficients[1, 5],
    coef = summary_cox$coefficients[1, 1],
    se = summary_cox$coefficients[1, 3],
    z = summary_cox$coefficients[1, 4]
  ))
  
  cat(sprintf("  %s: HR=%.3f (%.3f-%.3f), p=%.4f\n", 
              gene, 
              summary_cox$conf.int[1, 1],
              summary_cox$conf.int[1, 3],
              summary_cox$conf.int[1, 4],
              summary_cox$coefficients[1, 5]))
}

# 保存Cox回归结果
write.csv(cox_results, file.path(output_dir, "cox_univariate_results.csv"), row.names = FALSE)
cat("\n  已保存:", file.path(output_dir, "cox_univariate_results.csv"), "\n")

cat("\n=== 数据准备完成 ===\n")
cat("输出文件:\n")
cat("  1. adni_gene_expression.csv - 基因表达数据\n")
cat("  2. adni_survival_data.csv - 生存分析数据集\n")
cat("  3. gene_expression_stats.csv - 基因表达统计\n")
cat("  4. clinical_stats.csv - 临床特征统计\n")
cat("  5. cox_univariate_results.csv - Cox回归结果\n")
