# fig7 — External validation: iNPH signature reproduced in AD (ggplot2)
# 7a tissue×signature AUC heatmap | 7b co-expression of iNPH edges in AD
# 7c brain-region AUC | 7d brain vs blood specificity (negative control)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig7"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)

# ── fig7a: tissue x signature AUC heatmap ────────────────────────────────
a <- read.csv(PROC("figure7","fig7a"))
p7a <- ggplot(a, aes(signature, tissue)) +
  geom_tile(aes(fill = auc), colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.2f", auc)), size = FS_VALUE_MM) +
  scale_fill_gradient(low = PAL$transcriptomics, high = PAL$ad,
                      limits = c(0.5, 1), name = "AUC") +
  labs(x = NULL, y = NULL) +
  theme_pub(12) +
  theme(axis.line = element_blank(), axis.ticks = element_blank())
save_pub(p7a, OUT, "fig7a", w_in = 10, h_in = 5)

# ── fig7b: iNPH edges reproduced as co-expression in AD brain ─────────────
# NEW: network-level reproduction (statistically robust), replacing the
# flawed logFC-direction scatter (underlying Pearson r ≈ -0.06). For each
# measurable iNPH cross-tissue edge, Pearson co-expression r in AD vs Control.
b <- read.csv(PROC("figure7","fig7b"))
b$edge <- factor(b$edge, levels = b$edge)
bl <- b %>%
  select(edge, ad_r, ctrl_r) %>%
  pivot_longer(c(ad_r, ctrl_r), names_to = "group", values_to = "r") %>%
  mutate(group = recode(group, ad_r = "AD", ctrl_r = "Control"),
         group = factor(group, levels = c("AD", "Control")))
p7b <- ggplot(bl, aes(edge, r, fill = group)) +
  geom_hline(yintercept = 0, colour = "grey80") +
  geom_col(width = 0.72, colour = "black", linewidth = 0.35,
           position = position_dodge(width = 0.8)) +
  geom_text(aes(label = sprintf("%.2f", r), y = r + 0.03 * sign(r)),
            position = position_dodge(width = 0.8),
            vjust = ifelse(bl$r >= 0, 0, 1),
            size = FS_SMALL_MM, colour = PAL$darkgrey) +
  scale_fill_manual(values = c("AD" = PAL$ad, "Control" = PAL$nonsig), name = NULL) +
  labs(x = "iNPH cross-tissue edge", y = "Co-expression Pearson r in AD brain") +
  coord_cartesian(ylim = c(-0.2, 0.9)) +
  theme_pub(12) +
  theme(legend.position = "top",
        axis.text.x = element_text(angle = 25, hjust = 1))
save_pub(p7b, OUT, "fig7b", w_in = 10, h_in = 5)

# ── fig7c: iNPH brain signature AUC across brain regions ─────────────────
# NEW: region-level discriminative power (robust ROC-AUC), replacing the
# flawed signal-dose concordance curve. Brain / Blood / Full signatures x
# 3 brain regions, with bootstrap CI and permutation significance.
cc <- read.csv(PROC("figure7","fig7c"))
cc$tissue <- factor(cc$tissue, levels = c("All brain", "BA9", "Entorhinal"))
cc$sig_label <- ifelse(cc$perm_p < 0.05, "*", "")
p7c <- ggplot(cc, aes(tissue, auc, colour = signature, group = signature)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey70") +
  geom_line(linewidth = 0.7, position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.12,
                linewidth = 0.5, position = position_dodge(width = 0.5)) +
  geom_point(size = 2.8, position = position_dodge(width = 0.5)) +
  geom_text(aes(label = sig_label),
            vjust = -1.6, size = 7, fontface = "bold",
            position = position_dodge(width = 0.5), show.legend = FALSE) +
  scale_colour_manual(values = c(Brain = PAL$brain, Blood = PAL$blood,
                                 Full = PAL$metabolomics), name = NULL) +
  labs(x = NULL, y = "ROC-AUC (AD vs Control)") +
  coord_cartesian(ylim = c(0.45, 0.95)) +
  theme_pub(12) +
  theme(legend.position = "top")
save_pub(p7c, OUT, "fig7c", w_in = 10, h_in = 5)

# ── fig7d: brain-specificity — brain (validated) vs blood (negative ctrl) ─
# NEW: positive-vs-negative control contrast, replacing the flawed multi-
# stratum direction concordance. Full iNPH signature discriminates AD in
# brain (AUC>0.67, sig.) but NOT in PBMC (AUC~0.51, n.s.) — brain-specific.
d <- read.csv(PROC("figure7","fig7d"))
d$tissue <- factor(d$tissue, levels = d$tissue)
p7d <- ggplot(d, aes(tissue, auc, fill = group)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey70") +
  geom_col(width = 0.6, colour = "black", linewidth = 0.4) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.16,
                linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", auc), y = ci_upper + 0.025),
            vjust = 0, size = FS_VALUE_MM, fontface = "bold") +
  geom_text(aes(label = ifelse(perm_p < 0.05,
                   sprintf("p=%.3f *", perm_p), sprintf("p=%.2f ns", perm_p)),
                y = 0.04),
            colour = ifelse(d$perm_p < 0.05, PAL$ad, "grey50"),
            vjust = 0, size = FS_SMALL_MM) +
  scale_fill_manual(values = c(Brain = PAL$ad, Blood = PAL$nonsig), guide = "none") +
  labs(x = NULL, y = "ROC-AUC (Full iNPH signature)") +
  coord_cartesian(ylim = c(0, 0.9)) +
  theme_pub(12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_pub(p7d, OUT, "fig7d", w_in = 10, h_in = 5)

cat("FIG7 DONE\n")
