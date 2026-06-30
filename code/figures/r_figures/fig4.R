# fig4 — Dual-method virtual knockout validation (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig4"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)

# ── fig4a: coexpression validation dumbbell (Control vs AD r) ────────────
a <- read.csv(PROC("figure4","fig4a"))
a <- a %>% mutate(edge = paste(source, "\u2192", target),
                  edge = factor(edge, levels = edge[order(r_diff)]))
a_l <- a %>% select(edge, ctrl_r, ad_r, significant) %>%
  pivot_longer(c(ctrl_r, ad_r), names_to = "group", values_to = "r") %>%
  mutate(group = recode(group, ctrl_r = "Control", ad_r = "AD"),
         group = factor(group, c("Control","AD")))
# per-edge direction: reinforced in AD (r_diff>0, red) or attenuated (blue)
a_dir <- a %>% select(edge, r_diff) %>% distinct()
p4a <- ggplot(a_l, aes(r, edge)) +
  geom_segment(data = a %>% select(edge, ctrl_r, ad_r, r_diff) %>%
                 mutate(edge = factor(edge, levels = levels(a$edge))),
               aes(x = ctrl_r, xend = ad_r, y = edge, yend = edge,
                   colour = r_diff > 0), linewidth = 1.0, alpha = 0.55,
               show.legend = FALSE) +
  geom_point(aes(fill = group), shape = 21, size = 3.2, stroke = 0.5,
             colour = "black") +
  geom_vline(xintercept = 0, colour = "grey70", linetype = "dashed",
             linewidth = 0.4) +
  scale_colour_manual(values = c(`TRUE` = PAL$ad, `FALSE` = PAL$control)) +
  scale_fill_manual(values = c(Control = PAL$control, AD = PAL$ad), name = NULL) +
  labs(x = "Coexpression Pearson r", y = "Gene pair (source \u2192 target)") +
  theme_pub() +
  theme(axis.text.y = element_text(size = FS_SMALL),
        legend.position = "bottom")
save_pub(p4a, OUT, "fig4a", w_in = 10, h_in = 5)

# ── fig4b: SNHG5 KO Cohen's d lollipop (14 significant targets) ──────────
b <- read.csv(PROC("figure4","fig4b"))
b <- b %>% mutate(target_gene = fct_reorder(target_gene, cohens_d),
                  top2 = rank(-cohens_d, ties.method = "first") <= 2)
p4b <- ggplot(b, aes(cohens_d, target_gene)) +
  geom_segment(aes(x = 0, xend = cohens_d,
                   y = target_gene, yend = target_gene),
               colour = PAL$transcriptomics, linewidth = 0.9) +
  geom_point(aes(colour = top2), size = 3) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = 0.8, colour = "grey70", linetype = "dotted",
             linewidth = 0.4) +
  scale_colour_manual(values = c(`TRUE` = PAL$ad, `FALSE` = PAL$transcriptomics),
                      labels = c(`TRUE` = "Top-2 effect", `FALSE` = "Other"),
                      name = NULL) +
  labs(x = "Cohen's d", y = NULL) +
  theme_pub() +
  theme(axis.text.y = element_text(
          face = ifelse(order(order(b$cohens_d)) <= 2, "bold", "plain")),
        legend.position = "bottom")
save_pub(p4b, OUT, "fig4b", w_in = 10, h_in = 5)

# ── fig4c: validated virtual-knockout effects (9 genes, 22 events) ───────
# Only genes whose KO produced significant downstream effects are plotted
# (forward = hub knocked out; reverse = target knocked out). The 58 reverse
# genes with no effect are controls and are intentionally omitted.
c <- read.csv(PROC("figure4","fig4c"))
c <- c %>% mutate(
  gene = reorder(gene, n_significant),
  direction = factor(direction, c("Forward","Reverse")),
  omics = tools::toTitleCase(omics))
p4c <- ggplot(c, aes(n_significant, gene, fill = direction)) +
  geom_col(width = 0.65, colour = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0(n_significant, "/", n_targets)),
            hjust = -0.15, size = FS_VALUE_MM) +
  scale_fill_manual(values = c(Forward = PAL$ad, Reverse = PAL$transcriptomics),
                    name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(x = "Significant downstream targets", y = NULL) +
  theme_pub() +
  theme(legend.position = "top",
        axis.text.y = element_text(size = FS_AXIS))
save_pub(p4c, OUT, "fig4c", w_in = 10, h_in = 5)

# ── fig4d: negative-control comparison (KL divergence) ──────────────────
# Per-KO shaded control band (mean ± 1 SD of control KL) behind the target
# points, so each target sits visually above/beyond its own null band.
d <- read.csv(PROC("figure4","fig4d"))
d <- d %>% mutate(target_gene = factor(target_gene,
                              levels = target_gene[order(target_kl)]),
                  ko = paste0(ko_gene, " KO"),
                  sig = ifelse(significant, "p < 0.05", "n.s."))
# control band per KO (mean ± sd)
band <- d %>% group_by(ko) %>%
  summarise(lo = min(control_kl_mean - control_kl_std),
            hi = max(control_kl_mean + control_kl_std), .groups = "drop")
kocol <- c("SNHG5 KO" = PAL$transcriptomics,
           "PRKAR2B KO" = PAL$proteomics)
p4d <- ggplot(d, aes(target_kl, target_gene, colour = ko)) +
  geom_rect(data = band, aes(xmin = lo, xmax = hi, ymin = -Inf, ymax = Inf,
                             fill = ko), alpha = 0.18, colour = NA,
            inherit.aes = FALSE) +
  geom_point(aes(shape = sig), size = 3) +
  facet_grid(ko ~ ., scales = "free_y", space = "free_y") +
  scale_x_log10() +
  scale_colour_manual(values = kocol, guide = "none") +
  scale_fill_manual(values = kocol, guide = "none") +
  scale_shape_manual(values = c(`p < 0.05` = 18, `n.s.` = 16), name = NULL) +
  labs(x = "KL Divergence", y = NULL) +
  theme_pub() +
  theme(strip.text = element_text(size = FS_LEGEND, face = "bold"),
        legend.position = "bottom")
save_pub(p4d, OUT, "fig4d", w_in = 10, h_in = 5)

cat("FIG4 DONE\n")
