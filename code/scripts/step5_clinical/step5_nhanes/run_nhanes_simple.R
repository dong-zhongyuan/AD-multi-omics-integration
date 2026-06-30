#!/usr/bin/env Rscript

# ============================================================================
# NHANES分析 - Step3基因的生物标志物验证（简化版）
# ============================================================================
# 动态读取Step3输出的基因列表
# 数据: NHANES 2017-2018
# 分析: 通用生物标志物与认知功能关联
# ============================================================================

library(nhanesA)

# 加载项目配置
library(yaml)
config <- yaml::read_yaml("config/project_config.yaml")
PROJECT_ROOT <- config$paths$project_root
DATA_DIR <- file.path(PROJECT_ROOT, "data")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")
PROCESSED_DATA_DIR <- file.path(PROJECT_ROOT, "processed-data")

library(dplyr)
library(ggplot2)
library(tidyr)
library(broom)

# 加载Step5工具函数
source(file.path(PROJECT_ROOT, "scripts/step5_clinical/utils/step5_utils.R"))

output_dir <- file.path(OUTPUT_DIR, "step5_clinical_validation/nhanes_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

cat("================================================================================\n")
cat("NHANES分析 - Step3基因 (2017-2018周期)\n")
cat("================================================================================\n\n")

# 动态读取基因列表
print_gene_summary()
candidate_genes <- get_gene_symbols("all")

cat(sprintf("\n分析基因数: %d\n", length(candidate_genes)))
cat(sprintf("基因列表: %s\n\n", paste(candidate_genes, collapse=", ")))

cat("注意: 本分析使用通用生物标志物集合\n")
cat("      （白细胞、淋巴细胞、CRP、血红蛋白、红细胞等）\n\n")

# ============================================================================
# 1. 下载NHANES 2017-2018数据
# ============================================================================
cat("[Step 1] 下载NHANES 2017-2018数据...\n\n")

# 人口统计学
demo <- nhanes("DEMO_J")
cat("✅ 人口统计学数据: ", nrow(demo), "行\n")

# 认知功能测试
cog <- nhanes("CFQ_J")  
cat("✅ 认知功能数据: ", nrow(cog), "行\n")

# 血常规
cbc <- nhanes("CBC_J")
cat("✅ 血常规数据: ", nrow(cbc), "行\n")

# CRP
crp <- nhanes("HSCRP_J")
cat("✅ CRP数据: ", nrow(crp), "行\n\n")

# ============================================================================
# 2. 数据合并与清洗
# ============================================================================
cat("[Step 2] 数据合并与清洗...\n\n")

# 合并数据
complete_data <- demo %>%
  left_join(cog, by = "SEQN") %>%
  left_join(cbc, by = "SEQN") %>%
  left_join(crp, by = "SEQN") %>%
  filter(RIDAGEYR >= 60) %>%  # 60岁以上
  select(
    SEQN,
    age = RIDAGEYR,
    sex = RIAGENDR,
    race = RIDRETH3,
    # 认知功能
    cerad_immediate = CFDCST1,
    cerad_delayed = CFDCSR,
    animal_fluency = CFDAST,
    digit_symbol = CFDDS,
    # 生物标志物
    wbc = LBXWBCSI,           # 白细胞
    lymphocyte_pct = LBXLYPCT, # 淋巴细胞百分比
    hemoglobin = LBXHGB,       # 血红蛋白
    rbc = LBXRBCSI,            # 红细胞
    crp = LBXHSCRP             # CRP
  ) %>%
  mutate(
    sex_label = ifelse(sex == 1, "Male", "Female"),
    # 计算认知综合评分（标准化后平均）
    cog_composite = rowMeans(
      cbind(
        scale(cerad_immediate),
        scale(cerad_delayed),
        scale(animal_fluency),
        scale(digit_symbol)
      ),
      na.rm = TRUE
    )
  )

cat("合并后数据: ", nrow(complete_data), "行\n")
cat("60岁以上样本: ", nrow(complete_data), "人\n\n")

# ============================================================================
# 3. 描述性统计
# ============================================================================
cat("[Step 3] 描述性统计...\n\n")

# 人口统计学
demo_stats <- complete_data %>%
  summarise(
    n = n(),
    age_mean = mean(age, na.rm=TRUE),
    age_sd = sd(age, na.rm=TRUE),
    female_pct = sum(sex == 2, na.rm=TRUE) / n() * 100
  )

cat("人口统计学:\n")
cat(sprintf("  样本数: %d\n", demo_stats$n))
cat(sprintf("  年龄: %.1f ± %.1f 岁\n", demo_stats$age_mean, demo_stats$age_sd))
cat(sprintf("  女性: %.1f%%\n\n", demo_stats$female_pct))

# 生物标志物
biomarker_stats <- complete_data %>%
  summarise(
    wbc_mean = mean(wbc, na.rm=TRUE),
    wbc_sd = sd(wbc, na.rm=TRUE),
    lymphocyte_mean = mean(lymphocyte_pct, na.rm=TRUE),
    lymphocyte_sd = sd(lymphocyte_pct, na.rm=TRUE),
    hemoglobin_mean = mean(hemoglobin, na.rm=TRUE),
    hemoglobin_sd = sd(hemoglobin, na.rm=TRUE),
    rbc_mean = mean(rbc, na.rm=TRUE),
    rbc_sd = sd(rbc, na.rm=TRUE),
    crp_median = median(crp, na.rm=TRUE),
    crp_iqr = IQR(crp, na.rm=TRUE)
  )

cat("生物标志物水平:\n")
cat(sprintf("  WBC: %.2f ± %.2f (10^9/L)\n", biomarker_stats$wbc_mean, biomarker_stats$wbc_sd))
cat(sprintf("  淋巴细胞%%: %.1f ± %.1f\n", biomarker_stats$lymphocyte_mean, biomarker_stats$lymphocyte_sd))
cat(sprintf("  血红蛋白: %.1f ± %.1f (g/dL)\n", biomarker_stats$hemoglobin_mean, biomarker_stats$hemoglobin_sd))
cat(sprintf("  RBC: %.2f ± %.2f (10^12/L)\n", biomarker_stats$rbc_mean, biomarker_stats$rbc_sd))
cat(sprintf("  CRP: %.2f [IQR: %.2f] (mg/L)\n\n", biomarker_stats$crp_median, biomarker_stats$crp_iqr))

# ============================================================================
# 4. 生物标志物与认知功能关联
# ============================================================================
cat("[Step 4] 生物标志物与认知功能关联分析...\n\n")

biomarkers <- c("wbc", "lymphocyte_pct", "hemoglobin", "rbc", "crp")
biomarker_labels <- c("WBC", "Lymphocyte %", "Hemoglobin", "RBC", "CRP")

association_results <- data.frame()

for (i in 1:length(biomarkers)) {
  biomarker <- biomarkers[i]
  label <- biomarker_labels[i]
  
  # 线性回归：生物标志物 ~ 认知综合评分 + 年龄 + 性别
  formula_str <- paste0(biomarker, " ~ cog_composite + age + sex")
  model <- lm(as.formula(formula_str), data = complete_data)
  
  # 提取结果
  coef_summary <- summary(model)$coefficients
  
  if ("cog_composite" %in% rownames(coef_summary)) {
    association_results <- rbind(
      association_results,
      data.frame(
        biomarker = label,
        beta = coef_summary["cog_composite", "Estimate"],
        se = coef_summary["cog_composite", "Std. Error"],
        t_value = coef_summary["cog_composite", "t value"],
        p_value = coef_summary["cog_composite", "Pr(>|t|)"],
        n = nobs(model)
      )
    )
  }
}

cat("生物标志物与认知功能关联:\n")
print(association_results)
cat("\n")

write.csv(association_results, "biomarker_cognition_association.csv", row.names = FALSE)
cat("✅ 关联分析结果已保存: biomarker_cognition_association.csv\n\n")

# ============================================================================
# 5. 总结
# ============================================================================
cat("================================================================================\n")
cat("NHANES分析完成\n")
cat("================================================================================\n\n")

cat("分析内容:\n")
cat(sprintf("  1. %d个Step3基因\n", length(candidate_genes)))
cat("  2. NHANES 2017-2018数据 (60岁以上)\n")
cat("  3. 5个通用生物标志物与认知功能关联\n\n")

cat("输出文件:\n")
cat("  - biomarker_cognition_association.csv\n\n")

cat("关键发现:\n")
for (i in 1:nrow(association_results)) {
  biomarker <- association_results$biomarker[i]
  beta <- association_results$beta[i]
  p <- association_results$p_value[i]
  
  sig <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  
  cat(sprintf("  - %s: β=%.4f, p=%.4f %s\n", biomarker, beta, p, sig))
}

cat("\n注意: 本分析使用通用生物标志物，未针对特定基因\n")
cat("      如需基因特异性分析，需要基因表达数据或基因-生物标志物映射\n\n")

cat("================================================================================\n")
cat("分析完成！\n")
cat("================================================================================\n")
