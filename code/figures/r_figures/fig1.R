# fig1 — World model & pipeline overview (ggplot2)
source("theme_common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(forcats) })
FIG <- "fig1"; OUT <- file.path(ROOT, "figures", "r_figures", FIG)
JPG <- file.path(ROOT, "figures", "1a.jpg")

# ── fig1a: BioRender workflow (embed the pre-rendered 1a png) ────────────
# The raster is produced outside R (PIL converted 1a.jpg -> 1a.png at 300dpi);
# here we draw it onto a square canvas so the file set matches the others.
SRC_PNG <- file.path(ROOT, "figures", "1a.png")
if (!file.exists(SRC_PNG))
  SRC_PNG <- file.path(ROOT, "figures", "1a.jpg")
img <- png::readPNG(SRC_PNG, native = TRUE)
grob <- grid::rasterGrob(img, interpolate = TRUE)
p1a <- ggplot(data.frame()) + geom_blank() +
  annotation_custom(grob, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
  coord_fixed() +
  theme_void(base_family = "Arial")
save_pub(p1a, OUT, "fig1a", w_in = 10, h_in = 10)

# ── fig1b: OT distance / mapping quality by method (bar + errorbar) ──────
b <- read.csv(PROC("figure1","fig1b"))
p1b <- ggplot(b, aes(method, ot_distance_mean, fill = method)) +
  geom_col(width = 0.6, colour = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = ot_distance_mean - ot_distance_sd,
                    ymax = ot_distance_mean + ot_distance_sd), width = 0.2) +
  scale_fill_manual(values = setNames(
    c(PAL$transcriptomics, PAL$proteomics, PAL$metabolomics, PAL$ad, PAL$grey),
    b$method), guide = "none") +
  labs(x = NULL, y = "OT distance (brain \u2194 blood)") +
  theme_pub(12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_pub(p1b, OUT, "fig1b", w_in = 15, h_in = 5)

# ── fig1c: hub-rank eigengene decay by omics (3 lines) ───────────────────
c <- read.csv(PROC("figure1","fig1c"))
c <- c %>% mutate(omics = factor(omics,
              c("Transcriptomics","Proteomics","Metabolomics")))
p1c <- ggplot(c, aes(start_rank, mean_score, colour = omics)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.2) +
  scale_colour_manual(values = c(Transcriptomics = PAL$transcriptomics,
                                 Proteomics = PAL$proteomics,
                                 Metabolomics = PAL$metabolomics)) +
  labs(x = "Hub rank", y = "Eigengene score", colour = NULL) +
  theme_pub(12) + theme(legend.position = c(0.82, 0.85),
                        legend.background = element_rect(fill = "white",
                          colour = "grey80"))
save_pub(p1c, OUT, "fig1c", w_in = 10, h_in = 10)

# ── fig1d: validation loss convergence (relative scale) ─────────────────
# Only the validation curve is shown (per omics, normalised to each omics'
# first-epoch validation loss). This keeps the three omics comparable without
# amplifying the train/validation gap of the smaller metabolomics layer.
d <- read.csv(PROC("figure1","fig1d"))
d$omics <- factor(d$omics, c("Transcriptomics","Proteomics","Metabolomics"))
p1d <- ggplot(d, aes(epoch, val_loss_norm, colour = omics)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 1, linetype = "dotted", colour = "grey70") +
  scale_colour_manual(values = c(Transcriptomics = PAL$transcriptomics,
                                 Proteomics = PAL$proteomics,
                                 Metabolomics = PAL$metabolomics)) +
  labs(x = "Epoch", y = "Relative validation loss", colour = NULL) +
  theme_pub() + theme(legend.position = "top")
save_pub(p1d, OUT, "fig1d", w_in = 15, h_in = 5)

# ── fig1e: data scale by omics (features, samples, edges) ────────────────
# NOTE: fig1e_processed.csv stores n_samples and cross_tissue as Python dict
# reprs (e.g. "{'plasma': 1408, 'csf': 1408}"); parse the numbers out.
e <- read.csv(PROC("figure1","fig1e"), stringsAsFactors = FALSE)
e <- e %>% mutate(omics = factor(omics,
              c("Transcriptomics","Proteomics","Metabolomics")))
# extract plasma sample count and cross_tissue n_edges from the dict strings
e$n_plasma <- as.numeric(sub(".*plasma'?[: ]+([0-9]+).*", "\\1", e$n_samples))
e$n_edges  <- as.numeric(sub(".*n_edges'?[: ]+([0-9]+).*", "\\1", e$cross_tissue))
# manual reshape (3 series) to avoid tidyr version quirks
el <- rbind(
  data.frame(omics = e$omics, series = "Features",   count = e$n_features),
  data.frame(omics = e$omics, series = "Samples",    count = e$n_plasma),
  data.frame(omics = e$omics, series = "Cross-tissue edges", count = e$n_edges))
el$series <- factor(el$series, c("Features","Samples","Cross-tissue edges"))
p1e <- ggplot(el, aes(series, count, fill = omics)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(Transcriptomics = PAL$transcriptomics,
                               Proteomics = PAL$proteomics,
                               Metabolomics = PAL$metabolomics)) +
  scale_y_continuous(trans = "log10") +
  labs(x = NULL, y = "Count (log scale)", fill = NULL) +
  theme_pub() + theme(legend.position = "top",
                      axis.text.x = element_text(angle = 20, hjust = 1))
save_pub(p1e, OUT, "fig1e", w_in = 10, h_in = 5)

# ── fig1f: network topology radar (polar) ────────────────────────────────
f <- read.csv(PROC("figure1","fig1f"))
# normalised metrics for the radar
metrics <- c("n_edges_norm","mean_strength_norm",
             "mean_confidence_stability_norm","mean_confidence_snr_norm",
             "mean_confidence_consistency_norm")
labs_m <- c("Edges","Strength","Stability","SNR","Consistency")
f <- read.csv(PROC("figure1","fig1f"))
f$omics <- factor(f$omics, c("Transcriptomics","Proteomics","Metabolomics"))
# manual reshape to long (5 metrics x 4 omics rows)
fl <- do.call(rbind, lapply(seq_along(metrics), function(k) {
  data.frame(omics = f$omics, metric = labs_m[k], value = f[[metrics[k]]])
}))
fl$metric <- factor(fl$metric, labs_m)
p1f <- ggplot(fl, aes(metric, value, fill = omics)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(Transcriptomics = PAL$transcriptomics,
                               Proteomics = PAL$proteomics,
                               Metabolomics = PAL$metabolomics)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Topology metric", y = "Normalised score", fill = NULL) +
  theme_pub() + theme(legend.position = "top",
                      axis.text.x = element_text(angle = 15, hjust = 1))
save_pub(p1f, OUT, "fig1f", w_in = 10, h_in = 5)

cat("FIG1 DONE\n")
