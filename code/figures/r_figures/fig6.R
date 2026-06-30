# fig6 — Clinical translation: prognosis, targets, classification, age (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
library(ggforce)
FIG <- "fig6"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)

# ── fig6a: forest plot of hazard ratios ──────────────────────────────────
a <- read.csv(PROC("figure6","fig6a"))
a <- a %>% mutate(gene = fct_reorder(gene, HR),
                  sig = p_value < 0.05)
p6a <- ggplot(a, aes(HR, gene)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_errorbarh(aes(xmin = HR_lower, xmax = HR_upper,
                     colour = sig), height = 0.18, linewidth = 0.7) +
  geom_point(aes(colour = sig), shape = 18, size = 3.6) +
  scale_colour_manual(values = c(`TRUE` = PAL$ad, `FALSE` = PAL$nonsig),
                      labels = c(`TRUE` = "p < 0.05", `FALSE` = "n.s."),
                      name = NULL) +
  scale_x_continuous(limits = c(0.4, 2.2), breaks = c(0.5, 1, 1.5, 2)) +
  labs(x = "Hazard Ratio", y = NULL) +
  theme_pub() +
  theme(legend.position = c(0.8, 0.15),
        legend.background = element_rect(fill = "white", colour = "grey80"))
save_pub(p6a, OUT, "fig6a", w_in = 10, h_in = 5)

# ── fig6b: drug-target evidence dot matrix ───────────────────────────────
b <- read.csv(PROC("figure6","fig6b"))
# evidence axes: tractability, chembl molecules, dgidb drugs, max phase
ev <- b %>% transmute(
  symbol,
  `Tractability` = score_tractability,
  `ChEMBL mol.` = score_chembl,
  `DGIdb drugs` = score_dgidb,
  `Max phase` = score_phase) %>%
  pivot_longer(-symbol, names_to = "axis", values_to = "score") %>%
  mutate(symbol = factor(symbol, levels = b$symbol),
         axis = factor(axis, c("Tractability","ChEMBL mol.",
                               "DGIdb drugs","Max phase")))
p6b <- ggplot(ev, aes(axis, symbol, size = score, fill = score)) +
  geom_point(shape = 21, colour = "black", stroke = 0.5) +
  scale_size_continuous(range = c(1, 8), guide = "none") +
  scale_fill_gradient(low = "white", high = PAL$proteomics, guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_pub(12) +
  theme(panel.grid.major = element_line(colour = "grey92"))
save_pub(p6b, OUT, "fig6b", w_in = 10, h_in = 5)

# ── fig6c: gene classification parallel-sets (omics -> role) ─────────────
# Hand-built: axis 1 = Omics layer, axis 2 = Candidate role.
# Drop omics layers with zero genes so the diagram has no dead nodes.
c <- read.csv(PROC("figure6","fig6c"))
c$omics_label <- factor(c$omics_label,
                        c("Transcriptomics","Proteomics","Metabolomics"))
c$final_class <- factor(c$final_class,
                        c("Diagnostic","Both","Therapeutic"))
# keep only omics layers that actually have genes
keep_om <- names(which(table(c$omics_label) > 0))
c <- c[c$omics_label %in% keep_om, , drop = FALSE]
c$omics_label <- factor(c$omics_label, keep_om)
om_levels <- levels(c$omics_label)
cl_levels <- levels(c$final_class)

flow <- as.data.frame(table(c$omics_label, c$final_class))
colnames(flow) <- c("omics","role","n")
flow <- flow[flow$n > 0, ]
flow$omics <- factor(flow$omics, om_levels)
flow$role  <- factor(flow$role, cl_levels)

# stacked block (lo,hi) per axis, centered on 0
block_pos <- function(nvec) {
  cum <- cumsum(nvec); starts <- c(0, head(cum, -1))
  tot <- sum(nvec)
  data.frame(lo = starts - tot/2, hi = cum - tot/2)
}
om_pos <- block_pos(sapply(om_levels,
                           function(l) sum(flow$n[flow$omics == l])))
cl_pos <- block_pos(sapply(cl_levels,
                           function(l) sum(flow$n[flow$role  == l])))
rownames(om_pos) <- om_levels
rownames(cl_pos) <- cl_levels

# running cursor on each axis to match flows to blocks
cur_om <- setNames(rep(1, length(om_levels)), om_levels)
cur_cl <- setNames(rep(1, length(cl_levels)), cl_levels)
ribbons <- list()
for (i in seq_len(nrow(flow))) {
  o <- as.character(flow$omics[i]); r <- as.character(flow$role[i])
  oi <- cur_om[o]; ci <- cur_cl[r]
  y0_lo <- om_pos[o, "lo"][oi] + c(0, head(cumsum(flow$n[flow$omics==o]),-1))[oi]
  # simpler: compute block lo directly
  ob <- block_pos(flow$n[flow$omics == o])
  cb <- block_pos(flow$n[flow$role  == r])
  y0_lo <- ob$lo[oi]; y0_hi <- ob$hi[oi]
  y1_lo <- cb$lo[ci]; y1_hi <- cb$hi[ci]
  ribbons[[i]] <- data.frame(
    x = c(1, 1, 2, 2), y = c(y0_lo, y0_hi, y1_hi, y1_lo),
    id = i, fill = o)
  cur_om[o] <- oi + 1; cur_cl[r] <- ci + 1
}
rib <- do.call(rbind, ribbons)
om_col <- setNames(c(PAL$transcriptomics, PAL$proteomics, PAL$metabolomics),
                   om_levels)
p6c <- ggplot(rib, aes(x, y, group = id, fill = fill)) +
  geom_polygon(alpha = 0.5, colour = "white", linewidth = 0.3) +
  geom_rect(data = data.frame(
              x = 1, xmin = 0.9, xmax = 1.1,
              ymin = om_pos$lo, ymax = om_pos$hi,
              node = om_levels),
            aes(x = NULL, group = NA, fill = NULL,
                xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = om_col[om_levels], colour = "white", inherit.aes = FALSE) +
  geom_rect(data = data.frame(
              xmin = 1.9, xmax = 2.1,
              ymin = cl_pos$lo, ymax = cl_pos$hi,
              node = cl_levels),
            aes(x = NULL, group = NA, fill = NULL,
                xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = c(PAL$ad, PAL$teal, PAL$clinical), colour = "white",
            inherit.aes = FALSE) +
  geom_text(data = data.frame(x = 0.85, y = om_pos$lo + (om_pos$hi - om_pos$lo)/2,
                              lab = paste0(om_levels, " (n=",
                              sapply(om_levels, function(l) sum(flow$n[flow$omics==l])), ")")),
            aes(x = x, y = y, group = NA, fill = NULL, label = lab),
            hjust = 1, size = FS_VALUE_MM, inherit.aes = FALSE) +
  geom_text(data = data.frame(x = 2.15, y = cl_pos$lo + (cl_pos$hi - cl_pos$lo)/2,
                              lab = paste0(cl_levels, " (n=",
                              sapply(cl_levels, function(l) sum(flow$n[flow$role==l])), ")")),
            aes(x = x, y = y, group = NA, fill = NULL, label = lab),
            hjust = 0, size = FS_VALUE_MM, inherit.aes = FALSE) +
  scale_x_continuous(breaks = c(1, 2),
                     labels = c("Omics layer","Candidate role"),
                     limits = c(-1.2, 3.2)) +
  scale_fill_manual(values = om_col, name = NULL) +
  theme_void(base_family = "Arial", base_size = 13) +
  theme(legend.position = "bottom")
save_pub(p6c, OUT, "fig6c", w_in = 10, h_in = 5)

# ── fig6d: age-group immune/clinical markers (error bar lines) ────────────
d <- read.csv(PROC("figure6","fig6d"))
# manual reshape (avoid pivot_longer segfault on this R build)
mk <- c("Lymphocyte", "CRP", "WBC")
d_l <- rbind(
  data.frame(age_group = d$age_group, marker = "Lymphocyte",
             mean = d$lymphocyte_mean, sd = d$lymphocyte_sd),
  data.frame(age_group = d$age_group, marker = "CRP",
             mean = d$crp_mean, sd = d$crp_sd),
  data.frame(age_group = d$age_group, marker = "WBC",
             mean = d$wbc_mean, sd = d$wbc_sd))
d_l$marker <- factor(d_l$marker, mk)
d_l$age_group <- factor(d_l$age_group, c("60-69","70-79","80+"))
p6d <- ggplot(d_l, aes(age_group, mean, group = marker, colour = marker)) +
  geom_line(linewidth = 0.8) +
  geom_errorbar(aes(ymin = pmax(mean - sd, 0), ymax = mean + sd), width = 0.1,
                linewidth = 0.5) +
  geom_point(size = 2.4) +
  scale_colour_manual(values = c(Lymphocyte = PAL$transcriptomics,
                                 CRP = PAL$ad, WBC = PAL$teal)) +
  labs(x = "Age group", y = "Marker level", colour = NULL) +
  facet_wrap(~marker, scales = "free_y", nrow = 1) +
  theme_pub() +
  theme(legend.position = "none",
        strip.text = element_text(size = FS_LEGEND, face = "bold"))
save_pub(p6d, OUT, "fig6d", w_in = 10, h_in = 5)

cat("FIG6 DONE\n")
