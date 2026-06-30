# fig5 — Diagnostic model performance (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig5"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)

# ── fig5a: panel AUC comparison — efficiency lollipop ───────────────────
# marker SIZE encodes feature count, so the reader sees at a glance that
# Network-Guided (few features, high AUC) is far more efficient than
# Full Plasma (many features, marginally higher AUC).
a <- read.csv(PROC("figure5","fig5a"))
a <- a %>% mutate(
  panel_lab = recode(panel,
    full_plasma_l1 = "Full Plasma",
    network_guided_l1 = "Network-Guided",
    verified_proteomics = "Verified Proteomics",
    strict_forward = "Strict-Forward"),
  panel_lab = factor(panel_lab, c("Full Plasma","Network-Guided",
                                  "Verified Proteomics","Strict-Forward")),
  label = sprintf("%s (%d feat., AUC %.2f)", panel_lab, n_features,
                  cv_auc_mean),
  label = factor(label, levels = label[order(cv_auc_mean)]),
  hero = panel_lab == "Network-Guided")
p5a <- ggplot(a, aes(cv_auc_mean, label)) +
  geom_segment(aes(x = 0.5, xend = cv_auc_mean, y = label, yend = label,
                   colour = hero), linewidth = 0.9) +
  geom_errorbarh(aes(xmin = cv_auc_mean - cv_auc_sd,
                     xmax = cv_auc_mean + cv_auc_sd), height = 0.12,
                 colour = "grey40", linewidth = 0.5) +
  geom_point(aes(size = n_features, fill = panel_lab, shape = hero),
             colour = "black", stroke = 0.6) +
  geom_text(aes(x = cv_auc_mean, label = sprintf("%.2f", cv_auc_mean)),
            hjust = -0.9, size = FS_VALUE_MM) +
  scale_colour_manual(values = c(`TRUE` = PAL$ad, `FALSE` = "grey70"),
                      guide = "none") +
  scale_fill_manual(values = c(`Full Plasma` = PAL$purple,
                               `Network-Guided` = PAL$ad,
                               `Verified Proteomics` = PAL$teal,
                               `Strict-Forward` = PAL$grey), guide = "none") +
  scale_shape_manual(values = c(`TRUE` = 22, `FALSE` = 21), guide = "none") +
  scale_size_continuous(range = c(2.5, 14), name = "Features",
                        breaks = c(2, 16, 134)) +
  scale_x_continuous(limits = c(0.5, 1.02)) +
  labs(x = "Cross-validation AUC", y = NULL) +
  theme_pub() +
  theme(legend.position = c(0.82, 0.30),
        legend.background = element_rect(fill = "white", colour = "grey85"))
save_pub(p5a, OUT, "fig5a", w_in = 10, h_in = 5)

# ── fig5b: LASSO coefficients diverging bar (4 vs 22 features) ───────────
b <- read.csv(PROC("figure5","fig5b"))
# classify each feature into an AD-biology family for colour coding
fam <- function(f) {
  if (grepl("pTau|BD-pTau", f)) "Tau"
  else if (f %in% c("MAPT","AB42","AB40")) "Amyloid/protein"
  else if (f %in% c("TREM1","KLK6","IL7","BDNF","NPY","VEGFD")) "Neuro-immune"
  else "Other"
}
famcol <- c(Tau = PAL$ad, `Amyloid/protein` = PAL$purple,
            `Neuro-immune` = PAL$teal, Other = PAL$grey)
b <- b %>% mutate(
        family = sapply(feature, fam),
        panel = recode(panel,
          network_guided_l1 = "Network-Guided (4 features)",
          full_plasma_l1 = "Full Plasma (22 features)"),
        panel = factor(panel, c("Network-Guided (4 features)",
                                "Full Plasma (22 features)")),
        feature = reorder(feature, desc(coef)))
p5b <- ggplot(b, aes(coef, feature, fill = family)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = famcol, name = NULL) +
  facet_wrap(~panel, scales = "free_y", ncol = 2) +
  labs(x = "LASSO coefficient", y = NULL) +
  theme_pub() +
  theme(strip.text = element_text(size = FS_LEGEND, face = "bold"),
        axis.text.y = element_text(size = FS_SMALL),
        legend.position = "bottom")
save_pub(p5b, OUT, "fig5b", w_in = 10, h_in = 5)

# ── fig5c: cross-validation ROC curves ───────────────────────────────────
c <- read.csv(PROC("figure5","fig5c"))
# one curve per panel label; carry its AUC into the legend
auc_by <- c %>% group_by(label) %>%
  summarise(auc = round(auc[1], 2), nfeat = n_features[1], .groups = "drop")
auc_by$leg <- sprintf("%s (AUC %.2f, %d feat.)", auc_by$label, auc_by$auc,
                      auc_by$nfeat)
c <- c %>% left_join(auc_by %>% select(label, leg), by = "label")
pnlcol <- setNames(c(PAL$purple, PAL$ad, PAL$teal, PAL$grey), auc_by$leg)
p5c <- ggplot(c, aes(fpr, tpr, colour = leg)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey70") +
  geom_path(linewidth = 0.9) +
  scale_colour_manual(values = pnlcol) +
  labs(x = "False positive rate", y = "True positive rate", colour = NULL) +
  theme_pub() + theme(legend.position = c(0.72, 0.28),
                      legend.background = element_rect(fill = "white",
                        colour = "grey80"),
                      legend.text = element_text(size = FS_SMALL))
save_pub(p5c, OUT, "fig5c", w_in = 10, h_in = 5)

# ── fig5d: biomarker disease-correlation ranking ─────────────────────────
dd <- read.csv(PROC("figure5","fig5d"))
sel <- dd[dd$selected == TRUE | dd$selected == "True" | dd$selected == 1 |
          dd$selected == "1", , drop = FALSE]
p5d <- ggplot(dd, aes(rank, abs_correlation)) +
  geom_point(colour = "grey75", alpha = 0.6, size = 1) +
  geom_point(data = sel, colour = PAL$ad, size = 3) +
  ggrepel::geom_text_repel(data = sel, aes(label = gene),
                           colour = PAL$ad, size = FS_VALUE_MM, fontface = "bold",
                           nudge_y = 0.02, segment.size = 0.3,
                           min.segment.length = 0, max.overlaps = 20) +
  labs(x = "Biomarker rank (by |disease correlation|)",
       y = "|Pearson r| with diagnosis") +
  theme_pub(12)
save_pub(p5d, OUT, "fig5d", w_in = 10, h_in = 5)

cat("FIG5 DONE\n")
