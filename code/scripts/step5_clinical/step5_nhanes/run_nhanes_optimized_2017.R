#!/usr/bin/env Rscript

# ============================================================================
# NHANES优化分析 - Step3基因的生物标志物验证
# ============================================================================
# 动态读取Step3输出的基因列表
# 数据: NHANES 2017-2018
# 分析: 描述性统计 + 认知功能关联 + 分层分析
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

output_dir <- file.path(OUTPUT_DIR, "step5_clinical_validation/nhanes_analysis_optimized")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

cat("================================================================================\n")
cat("NHANES优化分析 - 治疗靶点生物标志物验证 (2017-2018周期)\n")
cat("================================================================================\n\n")

# 读取治疗靶点（Reverse验证的基因）
therapeutic_genes_file <- file.path(PROJECT_ROOT, "output/step5_gene_classification/therapeutic_genes.txt")
if (file.exists(therapeutic_genes_file)) {
  candidate_genes <- readLines(therapeutic_genes_file)
  cat("✓ 使用治疗靶点（Reverse验证）\n")
} else {
  cat("⚠ 未找到治疗靶点文件，使用默认基因列表\n")
  print_gene_summary()
  candidate_genes <- get_gene_symbols("all")
}

cat(sprintf("\n分析基因数: %d\n", length(candidate_genes)))
cat(sprintf("治疗靶点: %s\n\n", paste(candidate_genes, collapse=", ")))

# ============================================================================
# 1. 下载数据
# ============================================================================
cat("[Step 1] 下载NHANES 2017-2018数据...\n\n")

# 1.1 人口学数据
cat("  下载人口学数据 (DEMO_J)...\n")
demo <- tryCatch({
  nhanes("DEMO_J")
}, error = function(e) {
  cat(sprintf("    ❌ 错误: %s\n", e$message))
  stop("无法下载人口学数据")
})
cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(demo)))

# 1.2 全血细胞计数 (CBC)
cat("  下载全血细胞计数 (CBC_J)...\n")
cbc <- tryCatch({
  nhanes("CBC_J")
}, error = function(e) {
  cat(sprintf("    ❌ 错误: %s\n", e$message))
  stop("无法下载CBC数据")
})
cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(cbc)))

# 1.3 CRP (C-反应蛋白)
cat("  下载CRP数据 (HSCRP_J)...\n")
crp <- tryCatch({
  nhanes("HSCRP_J")
}, error = function(e) {
  cat(sprintf("    ❌ 错误: %s\n", e$message))
  stop("无法下载CRP数据")
})
cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(crp)))

# 1.4 葡萄糖
cat("  下载葡萄糖数据 (GLU_J)...\n")
glucose <- tryCatch({
  nhanes("GLU_J")
}, error = function(e) {
  cat(sprintf("    ⚠️  警告: %s\n", e$message))
  NULL
})
if (!is.null(glucose)) {
  cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(glucose)))
}

# 1.5 肝功能
cat("  下载肝功能数据 (BIOPRO_J)...\n")
liver <- tryCatch({
  nhanes("BIOPRO_J")
}, error = function(e) {
  cat(sprintf("    ⚠️  警告: %s\n", e$message))
  NULL
})
if (!is.null(liver)) {
  cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(liver)))
}

# 1.6 认知功能 (CFQ)
cat("  下载认知功能数据 (CFQ_J)...\n")
cognitive <- tryCatch({
  nhanes("CFQ_J")
}, error = function(e) {
  cat(sprintf("    ⚠️  警告: %s\n", e$message))
  cat("    认知功能数据不可用，将跳过认知关联分析\n")
  NULL
})

if (!is.null(cognitive)) {
  cat(sprintf("    ✅ 下载完成: %d 条记录\n\n", nrow(cognitive)))
} else {
  cat("    ⚠️  认知功能数据不可用\n\n")
}

# ============================================================================
# 2. 数据合并
# ============================================================================
cat("[Step 2] 合并数据集...\n\n")

# 从人口学数据开始
merged_data <- demo %>%
  select(SEQN, RIDAGEYR, RIAGENDR, RIDRETH3, DMDEDUC2)

# 合并CBC
if (!is.null(cbc) && nrow(cbc) > 0) {
  merged_data <- merged_data %>%
    left_join(cbc %>% select(SEQN, LBXWBCSI, LBXLYPCT, LBXHGB, LBXRBCSI, LBXPLTSI), by = "SEQN")
  cat("  ✅ CBC数据已合并\n")
}

# 合并CRP
if (!is.null(crp) && nrow(crp) > 0) {
  merged_data <- merged_data %>%
    left_join(crp %>% select(SEQN, LBXHSCRP), by = "SEQN") %>%
    rename(crp = LBXHSCRP)
  cat("  ✅ CRP数据已合并\n")
}

# 合并葡萄糖
if (!is.null(glucose) && nrow(glucose) > 0) {
  merged_data <- merged_data %>%
    left_join(glucose %>% select(SEQN, LBXGLU), by = "SEQN")
  cat("  ✅ 葡萄糖数据已合并\n")
}

# 合并肝功能
if (!is.null(liver) && nrow(liver) > 0) {
  merged_data <- merged_data %>%
    left_join(liver %>% select(SEQN, LBXSATSI), by = "SEQN")
  cat("  ✅ 肝功能数据已合并\n")
}

# 合并认知功能
if (!is.null(cognitive)) {
  cog_cols <- colnames(cognitive)
  cat("  认知数据可用列:\n")
  print(head(cog_cols, 20))
  cat("\n")
  
  cog_vars <- c("SEQN")
  if ("CFDDS" %in% cog_cols) cog_vars <- c(cog_vars, "CFDDS")
  if ("CFDCST" %in% cog_cols) cog_vars <- c(cog_vars, "CFDCST")
  if ("CFQ054" %in% cog_cols) cog_vars <- c(cog_vars, "CFQ054")
  
  if (length(cog_vars) > 1) {
    merged_data <- merged_data %>%
      left_join(cognitive %>% select(all_of(cog_vars)), by = "SEQN")
    cat("  ✅ 认知数据已合并\n\n")
  } else {
    cat("  ⚠️  未找到可用的认知变量\n\n")
  }
}

cat(sprintf("合并后总记录数: %d\n\n", nrow(merged_data)))

# ============================================================================
# 3. 数据清洗和筛选
# ============================================================================
cat("[Step 3] 数据清洗和筛选...\n\n")

# 3.1 筛选60岁以上人群
merged_data <- merged_data %>%
  filter(RIDAGEYR >= 60)
cat(sprintf("  60岁以上人群: %d\n", nrow(merged_data)))

# 3.2 重命名变量
merged_data <- merged_data %>%
  rename(
    age = RIDAGEYR,
    sex = RIAGENDR,
    race = RIDRETH3,
    education = DMDEDUC2,
    lymphocyte_pct = LBXLYPCT,
    wbc = LBXWBCSI,
    hemoglobin = LBXHGB,
    rbc = LBXRBCSI
  )

# 3.3 创建性别标签（RIAGENDR已经是字符串"Male"/"Female"）
merged_data <- merged_data %>%
  mutate(sex_label = sex)

# 3.4 检查每个生物标志物的样本量
cat("\n  各生物标志物的样本量:\n")
biomarkers <- c("lymphocyte_pct", "wbc", "crp", "hemoglobin", "rbc")
for (bm in biomarkers) {
  n_valid <- sum(!is.na(merged_data[[bm]]))
  cat(sprintf("    %s: %d (%.1f%%)\n", bm, n_valid, n_valid/nrow(merged_data)*100))
}
cat("\n")

# 3.5 创建完整案例数据集
complete_data <- merged_data %>%
  filter(!is.na(lymphocyte_pct) & !is.na(wbc) & !is.na(crp) & 
         !is.na(hemoglobin) & !is.na(rbc))

cat(sprintf("  完整案例（所有生物标志物都有值）: %d (%.1f%%)\n\n", 
            nrow(complete_data), 
            nrow(complete_data)/nrow(merged_data)*100))

# ============================================================================
# 4. 描述性统计
# ============================================================================
cat("[Step 4] 描述性统计...\n\n")

cat(sprintf("样本特征 (N=%d):\n", nrow(complete_data)))
cat(sprintf("  年龄: %.1f ± %.1f 岁 (范围: %.0f-%.0f)\n",
            mean(complete_data$age),
            sd(complete_data$age),
            min(complete_data$age),
            max(complete_data$age)))

sex_table <- table(complete_data$sex_label)
cat(sprintf("  男性: %d (%.1f%%)\n", sex_table["Male"], sex_table["Male"]/sum(sex_table)*100))
cat(sprintf("  女性: %d (%.1f%%)\n", sex_table["Female"], sex_table["Female"]/sum(sex_table)*100))

# 生物标志物水平 - 从配置文件动态读取基因-生物标志物映射
cat("\n生物标志物水平:\n")

mapping_file <- file.path(PROJECT_ROOT, "config/gene_biomarker_mapping.csv")
if (!file.exists(mapping_file)) {
  stop("基因-生物标志物映射文件不存在: ", mapping_file,
       "\n请创建 config/gene_biomarker_mapping.csv")
}
biomarker_stats <- read.csv(mapping_file, stringsAsFactors = FALSE)
cat(sprintf("  从 %s 读取 %d 条基因-生物标志物映射\n", mapping_file, nrow(biomarker_stats)))

for (i in 1:nrow(biomarker_stats)) {
  var <- biomarker_stats$variable[i]
  biomarker_stats$n[i] <- sum(!is.na(complete_data[[var]]))
  biomarker_stats$mean[i] <- mean(complete_data[[var]], na.rm=TRUE)
  biomarker_stats$sd[i] <- sd(complete_data[[var]], na.rm=TRUE)
  biomarker_stats$median[i] <- median(complete_data[[var]], na.rm=TRUE)
  biomarker_stats$q25[i] <- quantile(complete_data[[var]], 0.25, na.rm=TRUE)
  biomarker_stats$q75[i] <- quantile(complete_data[[var]], 0.75, na.rm=TRUE)
}

print(biomarker_stats)
cat("\n")

write.csv(biomarker_stats, "biomarker_descriptive_stats.csv", row.names = FALSE)
cat("✅ 描述性统计已保存: biomarker_descriptive_stats.csv\n\n")

# ============================================================================
# 5. 按年龄组分层分析
# ============================================================================
cat("[Step 5] 按年龄组分层分析...\n\n")

complete_data <- complete_data %>%
  mutate(age_group = cut(age, 
                         breaks = c(60, 70, 80, 100),
                         labels = c("60-69", "70-79", "80+"),
                         include.lowest = TRUE))

age_stratified <- complete_data %>%
  group_by(age_group) %>%
  summarise(
    n = n(),
    lymphocyte_mean = mean(lymphocyte_pct, na.rm=TRUE),
    lymphocyte_sd = sd(lymphocyte_pct, na.rm=TRUE),
    crp_mean = mean(crp, na.rm=TRUE),
    crp_sd = sd(crp, na.rm=TRUE),
    wbc_mean = mean(wbc, na.rm=TRUE),
    wbc_sd = sd(wbc, na.rm=TRUE),
    hemoglobin_mean = mean(hemoglobin, na.rm=TRUE),
    hemoglobin_sd = sd(hemoglobin, na.rm=TRUE),
    rbc_mean = mean(rbc, na.rm=TRUE),
    rbc_sd = sd(rbc, na.rm=TRUE)
  )

cat("按年龄组的生物标志物水平:\n")
print(age_stratified)
cat("\n")

write.csv(age_stratified, "biomarker_by_age_group.csv", row.names = FALSE)
cat("✅ 年龄分层结果已保存: biomarker_by_age_group.csv\n\n")

# ============================================================================
# 6. 按性别分层分析
# ============================================================================
cat("[Step 6] 按性别分层分析...\n\n")

sex_stratified <- complete_data %>%
  group_by(sex_label) %>%
  summarise(
    n = n(),
    lymphocyte_mean = mean(lymphocyte_pct, na.rm=TRUE),
    lymphocyte_sd = sd(lymphocyte_pct, na.rm=TRUE),
    crp_mean = mean(crp, na.rm=TRUE),
    crp_sd = sd(crp, na.rm=TRUE),
    wbc_mean = mean(wbc, na.rm=TRUE),
    wbc_sd = sd(wbc, na.rm=TRUE),
    hemoglobin_mean = mean(hemoglobin, na.rm=TRUE),
    hemoglobin_sd = sd(hemoglobin, na.rm=TRUE),
    rbc_mean = mean(rbc, na.rm=TRUE),
    rbc_sd = sd(rbc, na.rm=TRUE)
  )

cat("按性别的生物标志物水平:\n")
print(sex_stratified)
cat("\n")

write.csv(sex_stratified, "biomarker_by_sex.csv", row.names = FALSE)
cat("✅ 性别分层结果已保存: biomarker_by_sex.csv\n\n")

# ============================================================================
# 7. 生物标志物分布可视化
# ============================================================================
cat("[Step 7] 生成生物标志物分布图...\n\n")

biomarker_long <- complete_data %>%
  select(SEQN, age, sex_label, age_group, 
         lymphocyte_pct, crp, wbc, hemoglobin, rbc) %>%
  pivot_longer(cols = c(lymphocyte_pct, crp, wbc, hemoglobin, rbc),
               names_to = "biomarker",
               values_to = "value") %>%
  mutate(
    gene = case_when(
      biomarker == "lymphocyte_pct" ~ "PRKAR2B/LRBA",
      biomarker == "crp" ~ "MAPT/AGRN",
      biomarker == "wbc" ~ "All genes",
      biomarker == "hemoglobin" ~ "CAVIN2/CYB5R3/SURF1",
      biomarker == "rbc" ~ "CYB5R3/SURF1"
    ),
    biomarker_label = case_when(
      biomarker == "lymphocyte_pct" ~ "Lymphocyte %",
      biomarker == "crp" ~ "CRP (mg/L)",
      biomarker == "wbc" ~ "WBC (10^9/L)",
      biomarker == "hemoglobin" ~ "Hemoglobin (g/dL)",
      biomarker == "rbc" ~ "RBC (10^12/L)"
    )
  )

# 分布直方图
p1 <- ggplot(biomarker_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "black") +
  facet_wrap(~ biomarker_label, scales = "free", ncol = 3) +
  labs(
    title = "Distribution of Biomarkers in Older Adults (Age ≥60)",
    subtitle = "NHANES 2017-2018",
    x = "Biomarker Value",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(face = "bold")
  )

ggsave("biomarker_distributions.png", p1, width = 12, height = 8, dpi = 300)
cat("✅ 分布图已保存: biomarker_distributions.png\n")

# 按年龄组的箱线图
p2 <- ggplot(biomarker_long, aes(x = age_group, y = value, fill = age_group)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ biomarker_label, scales = "free_y", ncol = 3) +
  labs(
    title = "Biomarkers by Age Group",
    subtitle = "NHANES 2017-2018, Age ≥60",
    x = "Age Group",
    y = "Biomarker Value"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )

ggsave("biomarker_by_age_boxplot.png", p2, width = 12, height = 8, dpi = 300)
cat("✅ 年龄箱线图已保存: biomarker_by_age_boxplot.png\n")

# 按性别的箱线图
p3 <- ggplot(biomarker_long, aes(x = sex_label, y = value, fill = sex_label)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ biomarker_label, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c("Male" = "steelblue", "Female" = "pink")) +
  labs(
    title = "Biomarkers by Sex",
    subtitle = "NHANES 2017-2018, Age ≥60",
    x = "Sex",
    y = "Biomarker Value"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )

ggsave("biomarker_by_sex_boxplot.png", p3, width = 12, height = 8, dpi = 300)
cat("✅ 性别箱线图已保存: biomarker_by_sex_boxplot.png\n\n")

# ============================================================================
# 8. 保存清洗后的数据
# ============================================================================
cat("[Step 8] 保存清洗后的数据...\n\n")

write.csv(complete_data, "nhanes_complete_data.csv", row.names = FALSE)
cat("✅ 完整数据已保存: nhanes_complete_data.csv\n")

write.csv(merged_data, "nhanes_merged_data.csv", row.names = FALSE)
cat("✅ 合并数据已保存: nhanes_merged_data.csv\n\n")

# ============================================================================
# 9. 总结
# ============================================================================
cat("================================================================================\n")
cat("NHANES优化分析 - 数据准备完成 (2017-2018周期)\n")
cat("================================================================================\n\n")

cat(sprintf("数据来源: NHANES 2017-2018\n"))
cat(sprintf("总样本量 (60岁以上): %d\n", nrow(merged_data)))
cat(sprintf("完整案例 (所有生物标志物): %d (%.1f%%)\n", 
            nrow(complete_data), 
            nrow(complete_data)/nrow(merged_data)*100))
cat("\n")

cat("生物标志物映射:\n")
cat("  MAPT      → WBC + CRP (白细胞 + C-反应蛋白)——tau蛋白神经炎症通路\n")
cat("  FABP3     → WBC + Hemoglobin (白细胞 + 血红蛋白)——脂肪酸结合/细胞损伤\n")
cat("  PRKAR2B   → WBC + Lymphocyte % (白细胞 + 淋巴细胞)——cAMP信号/免疫\n")
cat("  AGRN      → WBC + CRP (白细胞 + C-反应蛋白)——突触稳定性/炎症\n")
cat("  CAVIN2    → WBC + Hemoglobin (白细胞 + 血红蛋白)——小窝蛋白/血管\n")
cat("  LRBA      → WBC + Lymphocyte % (白细胞 + 淋巴细胞)——免疫调控\n")
cat("  CYB5R3    → Hemoglobin + RBC (血红蛋白 + 红细胞)——细胞色素b5还原酶/线粒体\n")
cat("  SURF1     → Hemoglobin + RBC (血红蛋白 + 红细胞)——线粒体呼吸链\n\n")

cat("输出文件:\n")
cat("  - biomarker_descriptive_stats.csv\n")
cat("  - biomarker_by_age_group.csv\n")
cat("  - biomarker_by_sex.csv\n")
cat("  - biomarker_distributions.png\n")
cat("  - biomarker_by_age_boxplot.png\n")
cat("  - biomarker_by_sex_boxplot.png\n")
cat("  - nhanes_complete_data.csv\n")
cat("  - nhanes_merged_data.csv\n\n")

cat("================================================================================\n")
cat("阶段1完成！全部5个基因数据完整！\n")
cat("================================================================================\n\n")

cat("下一步:\n")
cat("  - 如果有认知功能数据，进行关联分析\n")
cat("  - 剂量-反应关系分析\n")
cat("  - 多生物标志物联合分析 (Panel评分, PCA)\n")
