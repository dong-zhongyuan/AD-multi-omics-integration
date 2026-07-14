# panel_h_ridge_kl.R — per-KO target KL distribution, Transcriptomics only.
# Single-module ridge plot (1:1 canvas): one ridge per KO gene (top-N by median
# target KL). KO Target (blue) overlaps Neg Control (gray) so the perturbation
# shift is visible per KO. x-range derived from the actual data so peaks center.
suppressPackageStartupMessages({
  library(ggplot2); library(ggridges); library(dplyr); library(tidyr)
})

here <- os.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/src")
DATA <- file.path(dirname(here), "data")
OUT  <- file.path(dirname(here), "output")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

LETTER <- "h"
FILL_COL <- "#0072B2"   # Transcriptomics blue
df <- read.csv(file.path(DATA, "fig3_genki_violin.csv"))

# Transcriptomics only; keep KOs with enough targets for a distribution
df <- df[df$omics == "Transcriptomics", ]
cnt <- df %>% count(ko_gene) %>% filter(n >= 6)
df  <- df %>% inner_join(cnt %>% select(ko_gene), by = "ko_gene")
df$target_kl_log  <- pmin(pmax(df$target_kl_log,  -3), 14)
df$control_kl_log <- pmin(pmax(df$control_kl_log, -3), 14)

# top-N KOs by median target KL (most-perturbed first)
N_TOP <- 14
top <- df %>%
  group_by(ko_gene) %>%
  summarise(m = median(target_kl_log, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(m)) %>% slice_head(n = N_TOP)
df <- df %>% filter(ko_gene %in% top$ko_gene)

# order: lowest median at bottom, highest at top
top <- top[order(top$m), ]
df$ko_gene <- factor(df$ko_gene, levels = top$ko_gene)

# x-range from data + padding so peaks are centered
rng <- range(c(df$target_kl_log, df$control_kl_log), na.rm = TRUE)
pad <- diff(rng) * 0.10
x_range <- c(rng[1] - pad, rng[2] + pad)

long <- bind_rows(
  df %>% transmute(ko_gene, type = "KO Target",   kl_log = target_kl_log),
  df %>% transmute(ko_gene, type = "Neg Control", kl_log = control_kl_log)
)
PAL <- c("KO Target" = FILL_COL, "Neg Control" = "#BBBBBB")

p <- ggplot(long, aes(x = kl_log, y = ko_gene, fill = type, color = type)) +
  geom_density_ridges(alpha = 0.55, scale = 1.1, size = 0.4,
                      from = x_range[1], to = x_range[2]) +
  scale_fill_manual(values = PAL) +
  scale_color_manual(values = c("KO Target" = FILL_COL, "Neg Control" = "#999999"),
                     guide = "none") +
  coord_cartesian(xlim = x_range) +
  labs(x = expression(log[10](KL)), y = NULL) +
  theme_minimal(base_family = "sans", base_size = 15) +
  theme(plot.title = element_blank(),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(size = 13),
        axis.title.x = element_text(size = 14, face = "bold"),
        legend.position = "top", legend.title = element_blank(),
        legend.text = element_text(size = 13),
        panel.grid.major.y = element_blank())

for (ext in c("png","pdf"))
  ggsave(file.path(OUT, paste0(LETTER, ".", ext)), p,
         width = 6, height = 6, dpi = 300, units = "in", bg = "white")
cat("Saved", LETTER, "png/pdf\n")
