# fig3 — Hub gene identification & KO validation (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig3"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)

# ── fig3a: composite centrality top hub genes (horizontal bar) ───────────
a <- read.csv(PROC("figure3","fig3a"))
# top 20 hubs; flat fill (no gradient), highlight top-5 hubs in proteomics red
a <- a %>% arrange(desc(composite_score)) %>% slice_head(n = 20)
a <- a %>% mutate(gene = factor(gene, levels = gene[order(composite_score)]),
                  tier = ifelse(row_number() <= 5, "top5", "rest"))
p3a <- ggplot(a, aes(composite_score, gene, fill = tier)) +
  geom_col(width = 0.72, colour = "black", linewidth = 0.25) +
  scale_fill_manual(values = c(top5 = PAL$proteomics, rest = "grey80"),
                    guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Composite centrality score", y = NULL) +
  theme_pub() +
  theme(axis.text.y = element_text(size = FS_SMALL,
          face = c(rep("bold", 5), rep("plain", 15))))
save_pub(p3a, OUT, "fig3a", w_in = 10, h_in = 10)

# ── fig3b: KO perturbation footprint (tested vs significant) ─────────────
# Both bars SOLID (matching fig3a/3c/3e): tested bar = tissue colour, black
# outline; significant bar = significant red, black outline. Per-row fills keep
# the transcriptomics/proteomics distinction that the old white-fill relied on.
b <- read.csv(PROC("figure3","fig3b"))
b <- b %>% mutate(KO_gene = factor(KO_gene,
                          levels = KO_gene[order(n_significant_targets)]),
                  tested_fill = ifelse(tissue == "transcriptomics",
                                       PAL$transcriptomics, PAL$proteomics))
p3b <- ggplot(b, aes(y = KO_gene)) +
  geom_col(aes(x = n_target_genes), fill = b$tested_fill,
           width = 0.38, colour = "black", linewidth = 0.4,
           position = position_nudge(y = 0.21)) +
  geom_col(aes(x = n_significant_targets), fill = PAL$significant,
           width = 0.38, colour = "black", linewidth = 0.4,
           position = position_nudge(y = -0.21)) +
  geom_text(aes(x = n_target_genes + 0.8, label = n_target_genes),
            nudge_y = 0.21, hjust = 0, size = FS_VALUE_MM, colour = PAL$darkgrey) +
  geom_text(aes(x = n_significant_targets + 0.8, label = n_significant_targets),
            nudge_y = -0.21, hjust = 0, size = FS_VALUE_MM, fontface = "bold") +
  labs(x = "Downstream genes", y = NULL) +
  coord_cartesian(xlim = c(0, max(b$n_target_genes) * 1.25)) +
  theme_pub(12) +
  theme(legend.position = "bottom")
save_pub(p3b, OUT, "fig3b", w_in = 10, h_in = 5)

# ── fig3c: Geneformer similarity per KO gene (cross-gene ratio) ──────────
c <- read.csv(PROC("figure3","fig3c"))
c <- c %>% mutate(gene = factor(gene, levels = gene[order(cross_gene_ratio)]))
p3c <- ggplot(c, aes(cross_gene_ratio, gene, fill = gene)) +
  geom_col(width = 0.55, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = setNames(
    c(PAL$transcriptomics, PAL$proteomics, PAL$metabolomics), c$gene),
    guide = "none") +
  labs(x = "Cross-gene ratio", y = NULL) +
  theme_pub(13)
save_pub(p3c, OUT, "fig3c", w_in = 10, h_in = 10)

# ── fig3d: predicted score vs validation effect size (scatter) ───────────
d <- read.csv(PROC("figure3","fig3d"))
d <- d %>% mutate(sig = ifelse(validation_significant, "Validated", "Not sig"),
                  omics = tools::toTitleCase(omics))
p3d <- ggplot(d, aes(predicted_score, validation_effect_size,
                     colour = omics)) +
  geom_hline(yintercept = 0, colour = "grey75", linewidth = 0.3) +
  geom_smooth(method = "lm", se = FALSE, colour = "grey55",
              linetype = "dashed", linewidth = 0.5, show.legend = FALSE) +
  geom_point(aes(shape = sig), size = 2.6, alpha = 0.85) +
  scale_colour_manual(values = c(Transcriptomics = PAL$transcriptomics,
                                 Proteomics = PAL$proteomics)) +
  scale_shape_manual(values = c(Validated = 16, `Not sig` = 1), name = NULL) +
  labs(x = "Predicted edge score", y = "Validation effect size",
       colour = NULL) +
  theme_pub() + theme(legend.position = "right")
save_pub(p3d, OUT, "fig3d", w_in = 10, h_in = 10)

# ── fig3e: filtering funnel (edge count by step) ────────────────────────
e <- read.csv(PROC("figure3","fig3e"))
# rebuild short labels per step, avoiding the intersection glyph (font-box issue)
short <- c("Raw cross-tissue edges",
           "Confidence + strength filter",
           "Brain-end hubs (MultiXrank)",
           "Hub intersect disease (overlap)")
e$label <- paste0("Step ", e$step, ": ", short[e$step])
p3e <- ggplot(e, aes(step, log10(count + 1), fill = color)) +
  geom_col(width = 0.62, colour = "black", linewidth = 0.3) +
  geom_text(aes(label = comma(count)), vjust = -0.6, size = FS_VALUE_MM) +
  scale_fill_manual(values = c(C_TRANSCRIPTOMICS = PAL$transcriptomics,
                               C_PROTEOMICS = PAL$proteomics,
                               C_METABOLOMICS = PAL$metabolomics),
                    labels = c("Transcriptomics","Proteomics","Metabolomics"),
                    guide = "none") +
  scale_x_continuous(breaks = e$step, labels = e$label) +
  labs(x = NULL, y = expression(log[10](Edges))) +
  theme_pub(12) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = FS_AXIS))
save_pub(p3e, OUT, "fig3e", w_in = 10, h_in = 10)

# ── fig3f: predicted rank vs validation effect (scatter) ────────────────
f <- read.csv(PROC("figure3","fig3f"))
p3f <- ggplot(f, aes(predicted_rank, validation_effect_size,
                     colour = omics, size = -log10(validation_p_value))) +
  geom_point(alpha = 0.8) +
  scale_colour_manual(values = c(transcriptomics = PAL$transcriptomics,
                                 proteomics = PAL$proteomics)) +
  scale_size_continuous(range = c(1.5, 6), name = expression(-log[10]~p)) +
  labs(x = "Predicted rank", y = "Validation effect size", colour = NULL) +
  theme_pub(12) + theme(legend.position = "right")
save_pub(p3f, OUT, "fig3f", w_in = 10, h_in = 5)

cat("FIG3 DONE\n")
