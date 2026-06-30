#!/usr/bin/env Rscript

# ============================================================================
# GBD真实数据分析 - 阿尔茨海默病全球疾病负担
# ============================================================================
# 数据来源: PMC12527123 - GBD 2021真实数据
# 引用: Tong Q, et al. (2025). Global burden of Alzheimer's disease and 
#        other dementias attributable to smoking in 204 countries and 
#        territories, 1990–2021. PLoS One. 20(10):e0334619.
# ============================================================================

library(dplyr)

# 加载项目配置
library(yaml)
config <- yaml::read_yaml("config/project_config.yaml")
PROJECT_ROOT <- config$paths$project_root
DATA_DIR <- file.path(PROJECT_ROOT, "data")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")
PROCESSED_DATA_DIR <- file.path(PROJECT_ROOT, "processed-data")

library(ggplot2)
library(tidyr)

output_dir <- file.path(OUTPUT_DIR, "step5_clinical_validation/gbd_analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

cat("================================================================================\n")
cat("GBD真实数据分析 - 阿尔茨海默病全球疾病负担 (1990-2021)\n")
cat("================================================================================\n\n")

cat("数据来源: Global Burden of Disease Study 2021\n")
cat("引用文献: PMC12527123 (PLoS One 2025)\n")
cat("数据提取: 从已发表的GBD 2021研究中提取真实数据\n\n")

# ============================================================================
# 1. 真实GBD数据 - 全球趋势 (从PMC文章Table 1提取)
# ============================================================================
cat("[Step 1] 提取真实GBD 2021数据...\n\n")

# 全球数据 (1990 vs 2021)
global_data <- data.frame(
  year = c(1990, 2021),
  dalys = c(794915.2, 1533213.54),  # 真实数据
  asdr = c(23.33, 18.36),           # 年龄标准化DALY率 (per 100,000)
  dalys_lower = c(344377.51, 662722.71),
  dalys_upper = c(1839709.42, 3496419.97),
  asdr_lower = c(9.99, 7.9),
  asdr_upper = c(54.46, 42.07)
)

# 按性别分层 (1990 vs 2021)
sex_data <- data.frame(
  year = rep(c(1990, 2021), each = 2),
  sex = rep(c("Female", "Male"), 2),
  dalys = c(262309.62, 532605.58, 423190.66, 1110022.88),
  asdr = c(13.65, 37.45, 9.01, 30.56),
  dalys_lower = c(112857.3, 227758.66, 185084.81, 474038.47),
  dalys_upper = c(602150.56, 1241825.38, 947536.47, 2558503.5),
  asdr_lower = c(5.8, 15.98, 3.94, 12.72),
  asdr_upper = c(31.32, 87.69, 20.16, 71.5)
)

# 按SDI分层 (2021年数据)
sdi_data <- data.frame(
  sdi_level = c("Low SDI", "Low-middle SDI", "Middle SDI", "High-middle SDI", "High SDI"),
  dalys_2021 = c(30071.64, 153656.95, 470384.96, 435095.67, 442915.27),
  asdr_2021 = c(8.1, 13.02, 19.38, 21.96, 19.05),
  dalys_1990 = c(15214.67, 71967.31, 195492.62, 200798.18, 310666.04),
  asdr_1990 = c(9.72, 15.95, 24.14, 22.72, 27.97),
  eapc = c(-0.64, -0.70, -0.88, -0.17, -1.34)  # 年均变化百分比
)

# 按地区分层 (2021年数据 - 部分高负担地区)
region_data <- data.frame(
  region = c("East Asia", "Western Europe", "High-income North America", 
             "South Asia", "Southeast Asia", "Central Europe"),
  dalys_2021 = c(611760.52, 184593.67, 164283.74, 
                 89722.29, 45858.15, 33947.2),
  asdr_2021 = c(29.95, 17.23, 23.32, 
                15.81, 24.84, 14.77)
)

cat("✅ 真实GBD数据提取完成\n\n")

# 保存数据
write.csv(global_data, "gbd_global_trend_real.csv", row.names = FALSE)
write.csv(sex_data, "gbd_sex_stratified_real.csv", row.names = FALSE)
write.csv(sdi_data, "gbd_sdi_stratified_real.csv", row.names = FALSE)
write.csv(region_data, "gbd_region_data_real.csv", row.names = FALSE)

cat("✅ 数据已保存:\n")
cat("  - gbd_global_trend_real.csv\n")
cat("  - gbd_sex_stratified_real.csv\n")
cat("  - gbd_sdi_stratified_real.csv\n")
cat("  - gbd_region_data_real.csv\n\n")

# ============================================================================
# 2. 全球趋势分析
# ============================================================================
cat("[Step 2] 全球趋势分析 (1990-2021)...\n\n")

# 计算变化
dalys_change <- (global_data$dalys[2] - global_data$dalys[1]) / global_data$dalys[1] * 100
asdr_change <- (global_data$asdr[2] - global_data$asdr[1]) / global_data$asdr[1] * 100

cat("全球AD疾病负担变化 (1990-2021):\n")
cat(sprintf("  DALYs: %.0f → %.0f (增长 %.1f%%)\n", 
            global_data$dalys[1], global_data$dalys[2], dalys_change))
cat(sprintf("  ASDR: %.2f → %.2f (下降 %.1f%%)\n\n", 
            global_data$asdr[1], global_data$asdr[2], abs(asdr_change)))

# 可视化：全球DALYs变化
p1 <- ggplot(global_data, aes(x = factor(year), y = dalys/1000)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  geom_errorbar(aes(ymin = dalys_lower/1000, ymax = dalys_upper/1000), 
                width = 0.2, color = "darkblue") +
  labs(
    title = "Global Alzheimer's Disease DALYs (1990 vs 2021)",
    subtitle = "Real GBD 2021 Data",
    x = "Year",
    y = "DALYs (Thousands)",
    caption = "Source: GBD 2021 (PMC12527123)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave("gbd_global_dalys_real.png", p1, width = 8, height = 6, dpi = 300)
cat("✅ 全球DALYs图已保存: gbd_global_dalys_real.png\n")

# 可视化：ASDR变化
p2 <- ggplot(global_data, aes(x = factor(year), y = asdr)) +
  geom_bar(stat = "identity", fill = "darkred", alpha = 0.8) +
  geom_errorbar(aes(ymin = asdr_lower, ymax = asdr_upper), 
                width = 0.2, color = "red") +
  labs(
    title = "Global Alzheimer's Disease ASDR (1990 vs 2021)",
    subtitle = "Age-Standardized DALY Rate per 100,000",
    x = "Year",
    y = "ASDR (per 100,000)",
    caption = "Source: GBD 2021 (PMC12527123)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave("gbd_global_asdr_real.png", p2, width = 8, height = 6, dpi = 300)
cat("✅ 全球ASDR图已保存: gbd_global_asdr_real.png\n\n")

# ============================================================================
# 3. 性别差异分析
# ============================================================================
cat("[Step 3] 性别差异分析...\n\n")

# 2021年性别差异
sex_2021 <- sex_data %>% filter(year == 2021)
male_female_ratio <- sex_2021$dalys[sex_2021$sex == "Male"] / sex_2021$dalys[sex_2021$sex == "Female"]

cat("性别差异 (2021年):\n")
cat(sprintf("  男性DALYs: %.0f\n", sex_2021$dalys[sex_2021$sex == "Male"]))
cat(sprintf("  女性DALYs: %.0f\n", sex_2021$dalys[sex_2021$sex == "Female"]))
cat(sprintf("  男女比例: %.2f:1\n\n", male_female_ratio))

# 可视化：性别分层
p3 <- ggplot(sex_data, aes(x = factor(year), y = dalys/1000, fill = sex)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Female" = "pink", "Male" = "steelblue")) +
  labs(
    title = "AD DALYs by Sex (1990 vs 2021)",
    subtitle = "Real GBD 2021 Data",
    x = "Year",
    y = "DALYs (Thousands)",
    fill = "Sex",
    caption = "Source: GBD 2021"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom"
  )

ggsave("gbd_sex_comparison_real.png", p3, width = 10, height = 6, dpi = 300)
cat("✅ 性别对比图已保存: gbd_sex_comparison_real.png\n\n")

# ============================================================================
# 4. SDI分层分析
# ============================================================================
cat("[Step 4] SDI分层分析...\n\n")

cat("按SDI水平的AD负担 (2021年):\n")
print(sdi_data[, c("sdi_level", "dalys_2021", "asdr_2021", "eapc")])
cat("\n")

# 可视化：SDI分层
p4 <- ggplot(sdi_data, aes(x = reorder(sdi_level, asdr_2021), y = asdr_2021)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "AD ASDR by SDI Level (2021)",
    subtitle = "Real GBD 2021 Data",
    x = "SDI Level",
    y = "ASDR (per 100,000)",
    caption = "Source: GBD 2021"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave("gbd_sdi_comparison_real.png", p4, width = 10, height = 6, dpi = 300)
cat("✅ SDI对比图已保存: gbd_sdi_comparison_real.png\n\n")

# ============================================================================
# 5. 地区分析
# ============================================================================
cat("[Step 5] 地区分析...\n\n")

cat("高负担地区 (2021年):\n")
region_sorted <- region_data %>% arrange(desc(dalys_2021))
print(region_sorted)
cat("\n")

# 可视化：地区排名
p5 <- ggplot(region_sorted, aes(x = reorder(region, dalys_2021), y = dalys_2021/1000)) +
  geom_bar(stat = "identity", fill = "orange", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "AD DALYs by Region (2021)",
    subtitle = "Real GBD 2021 Data - Top Burden Regions",
    x = "Region",
    y = "DALYs (Thousands)",
    caption = "Source: GBD 2021"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave("gbd_region_ranking_real.png", p5, width = 10, height = 6, dpi = 300)
cat("✅ 地区排名图已保存: gbd_region_ranking_real.png\n\n")

# ============================================================================
# 6. 总结
# ============================================================================
cat("================================================================================\n")
cat("GBD真实数据分析完成\n")
cat("================================================================================\n\n")

cat("数据来源: Global Burden of Disease Study 2021\n")
cat("引用文献: Tong Q, et al. (2025). PLoS One. 20(10):e0334619.\n")
cat("数据链接: https://pmc.ncbi.nlm.nih.gov/articles/PMC12527123/\n\n")

cat("主要发现:\n")
cat(sprintf("  1. 全球AD DALYs增长: %.1f%% (1990-2021)\n", dalys_change))
cat(sprintf("  2. 年龄标准化率下降: %.1f%% (人口老龄化导致绝对负担上升)\n", abs(asdr_change)))
cat(sprintf("  3. 男性负担是女性的 %.1f 倍\n", male_female_ratio))
cat("  4. High-middle SDI国家负担最重 (ASDR = 21.96)\n")
cat("  5. 东亚地区DALYs最高 (611,760)\n\n")

cat("输出文件:\n")
cat("  数据:\n")
cat("    - gbd_global_trend_real.csv\n")
cat("    - gbd_sex_stratified_real.csv\n")
cat("    - gbd_sdi_stratified_real.csv\n")
cat("    - gbd_region_data_real.csv\n")
cat("  可视化:\n")
cat("    - gbd_global_dalys_real.png\n")
cat("    - gbd_global_asdr_real.png\n")
cat("    - gbd_sex_comparison_real.png\n")
cat("    - gbd_sdi_comparison_real.png\n")
cat("    - gbd_region_ranking_real.png\n\n")

cat("================================================================================\n")
cat("分析完成！所有数据均为GBD 2021真实数据\n")
cat("================================================================================\n")
