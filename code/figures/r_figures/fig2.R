# fig2 — Cross-Tissue Sensitivity Network Characterization (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig2"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)
PAL3 <- c(Transcriptomics = PAL$transcriptomics, Proteomics = PAL$proteomics,
          Metabolomics = PAL$metabolomics)

# ── fig2a: edge confidence boxplots (stability, snr, consistency) ────────
a <- read.csv(PROC("figure2","fig2a"))
a_l <- a %>% pivot_longer(c(confidence_stability, confidence_snr,
                            confidence_consistency),
                          names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
            confidence_stability = "Stability",
            confidence_snr = "Signal/noise",
            confidence_consistency = "Consistency"),
         metric = factor(metric, c("Stability","Signal/noise","Consistency")),
         omics = factor(omics, c("Transcriptomics","Proteomics","Metabolomics")))
# sample jitter points (boxplot stats computed on full data via separate geom;
# 2500 per omics keeps the SVG small without distorting the density)
set.seed(42)
a_jit <- a_l %>% group_by(omics, metric) %>%
  slice_sample(n = 2500) %>% ungroup()
p2a <- ggplot(a_l, aes(omics, value, fill = omics)) +
  geom_boxplot(outlier.shape = NA, linewidth = 0.4, key_glyph = "rect") +
  geom_jitter(data = a_jit, width = 0.18, alpha = 0.18, size = 0.5,
              aes(colour = omics)) +
  facet_wrap(~metric, nrow = 1) +
  scale_fill_manual(values = PAL3) +
  scale_colour_manual(values = PAL3) +
  labs(x = NULL, y = "Edge confidence") +
  theme_pub(13) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none")
save_pub(p2a, OUT, "fig2a", w_in = 10, h_in = 5)

# ── fig2b: top-15 proteomics cross-tissue edges (lollipop) ───────────────
b <- read.csv(PROC("figure2","fig2b"))
family <- function(e) {
  if (grepl("pTau|BD-pTau|MAPT", e)) "Tau / MAPT"
  else if (grepl("A\u03b2", e)) "Amyloid-\u03b2"
  else if (grepl("VEGFD|NPY|BDNF|IL7|PTN|pTDP43", e)) "Neuro-peptide"
  else "Other"
}
famcol <- c("Tau / MAPT" = PAL$proteomics, "Amyloid-\u03b2" = PAL$transcriptomics,
            "Neuro-peptide" = PAL$metabolomics, Other = "#888888")
b <- read.csv(PROC("figure2","fig2b"))
b <- b %>% mutate(family = sapply(edge, family),
                  edge = paste0(source, " - ", target),
                  edge = factor(edge, levels = edge[order(score)]))
p2b <- ggplot(b, aes(score, edge, colour = family)) +
  geom_segment(aes(x = 0, xend = score, y = edge, yend = edge),
               linewidth = 1.1, alpha = 0.85) +
  geom_point(aes(size = score), show.legend = FALSE) +
  scale_colour_manual(values = famcol, name = "AD-biology family") +
  scale_size_continuous(range = c(2.5, 4.5), guide = "none") +
  labs(x = "Cross-tissue edge score", y = NULL) +
  theme_pub() +
  theme(legend.position = "bottom",
        legend.box.spacing = unit(0.2, "cm"),
        axis.text.y = element_text(size = FS_SMALL))
save_pub(p2b, OUT, "fig2b", w_in = 10, h_in = 5)

# ── fig2c: network-scale fingerprint (2 panels) ──────────────────────────
# DODGED POINTS (not bars): bars don't render on a log axis (zero baseline =
# -Inf). Points at each value read cleanly on log scale.
c <- read.csv(PROC("figure2","fig2c"))
c <- c %>% mutate(layer = factor(layer,
              c("Transcriptomics","Proteomics","Metabolomics")))
cl <- rbind(
  data.frame(layer = c$layer, series = "Features",       count = c$n_features),
  data.frame(layer = c$layer, series = "Plasma samples", count = c$n_plasma_samples),
  data.frame(layer = c$layer, series = "CSF samples",    count = c$n_csf_samples))
cl$series <- factor(cl$series, c("Features","Plasma samples","CSF samples"))
p2cL <- ggplot(cl, aes(layer, count, colour = series)) +
  geom_point(position = position_dodge(width = 0.6), size = 3.2) +
  scale_y_log10(limits = c(20, 120000),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_colour_manual(values = c(PAL$transcriptomics, PAL$proteomics,
                                 PAL$metabolomics)) +
  labs(x = NULL, y = "Count (log scale)", colour = NULL) +
  theme_pub() + theme(legend.position = "top")

cr <- rbind(
  data.frame(layer = c$layer, series = "Brain-internal",  count = c$n_brain_edges),
  data.frame(layer = c$layer, series = "Blood-internal",  count = c$n_blood_edges),
  data.frame(layer = c$layer, series = "Cross-tissue",    count = c$n_cross_tissue_edges))
cr$series <- factor(cr$series, c("Brain-internal","Blood-internal","Cross-tissue"))
p2cR <- ggplot(cr, aes(layer, count, colour = series)) +
  geom_point(position = position_dodge(width = 0.6), size = 3.2) +
  scale_y_log10(limits = c(300, 1500000),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_colour_manual(values = c(PAL$transcriptomics, PAL$proteomics,
                                 PAL$metabolomics)) +
  labs(x = NULL, y = "Sensitivity edges (log scale)", colour = NULL) +
  theme_pub() + theme(legend.position = "top")
library(patchwork)
p2c <- (p2cL | p2cR) + plot_layout(widths = c(1, 1.15))
save_pub(p2c, OUT, "fig2c", w_in = 10, h_in = 5)

# ── fig2d: edge counts by directionality (bubble heatmap) ───────────────
d <- read.csv(PROC("figure2","fig2d"))
d <- d %>% mutate(omics = tools::toTitleCase(omics),
                  omics = factor(omics, c("Transcriptomics","Proteomics","Metabolomics")),
                  direction = factor(direction, c("Brain-internal","Blood-internal","Cross-tissue")),
                  bubble = log10(edge_count + 1))
lab_fmt <- function(v) ifelse(v >= 1000, paste0(round(v/1000), "k"), as.character(v))
p2d <- ggplot(d, aes(omics, direction, size = bubble, colour = omics)) +
  geom_point(shape = 21, fill = NA, stroke = 1.6) +
  geom_text(aes(label = lab_fmt(edge_count)), size = FS_VALUE_MM,
            colour = "black", fontface = "bold", show.legend = FALSE) +
  scale_colour_manual(values = PAL3, guide = "none") +
  scale_size_continuous(range = c(6, 24), guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_pub(14) +
  theme(panel.grid.major = element_line(colour = "grey85", linetype = "dashed"))
save_pub(p2d, OUT, "fig2d", w_in = 10, h_in = 5)

cat("FIG2 DONE\n")
