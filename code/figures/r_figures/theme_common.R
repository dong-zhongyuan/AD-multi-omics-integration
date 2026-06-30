# ── Shared infrastructure for all ggplot2 figures ─────────────────────────
# Mirrors figures/20260525/style.py palette so R versions match the
# Python originals. Source-load this from every panel script.

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

# project palette (hex identical to style.py)
PAL <- list(
  transcriptomics = "#0072B2",
  proteomics      = "#D55E00",
  metabolomics    = "#009E73",
  brain           = "#7B2D8B",
  blood           = "#E8590C",
  ad              = "#C62828",
  control         = "#1565C0",
  red             = "#E8655A",
  green           = "#6AB56E",
  blue            = "#7EAED3",
  purple          = "#9B7DB8",
  orange          = "#E8A952",
  teal            = "#0E8C6A",
  significant     = "#C62828",
  nonsig          = "#BDBDBD",
  approved        = "#2ECC71",
  clinical        = "#F1C40F",
  preclinical     = "#E67E22",
  grey            = "#777777",
  darkgrey        = "#444444"
)

# UNIFIED FONT-SIZE SCALE — every panel must use these.
# Single authoritative scale so the whole figure set reads as one family.
# NOTE: element_text size is in POINTS; geom_text size is in MILLIMETRES.
# Use FS_*_*MM for geom_text so the rendered pt size matches the pt scale.
PT_PER_MM <- 2.845
FS_TITLE  <- 16   # panel/axis titles (bold)
FS_AXIS   <- 14   # axis tick labels
FS_LEGEND <- 13   # legend entries
FS_VALUE  <- 12   # on-figure value labels (bar counts, AUC numbers)
FS_SMALL  <- 11   # dense tick labels (many genes, rotated x labels)
# geom_text/geom_label equivalents (mm = pt / 2.845)
FS_TITLE_MM  <- FS_TITLE  / PT_PER_MM   # 5.6
FS_AXIS_MM   <- FS_AXIS   / PT_PER_MM   # 4.9
FS_LEGEND_MM <- FS_LEGEND / PT_PER_MM   # 4.6
FS_VALUE_MM  <- FS_VALUE  / PT_PER_MM   # 4.2
FS_SMALL_MM  <- FS_SMALL  / PT_PER_MM   # 3.9

# publication theme: thin axes, white bg, Arial, unified font scale
theme_pub <- function(base_size = FS_AXIS) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line   = element_line(linewidth = 0.5, colour = "black"),
      axis.ticks  = element_line(linewidth = 0.5, colour = "black"),
      axis.title  = element_text(size = FS_TITLE, face = "bold"),
      axis.text   = element_text(size = FS_AXIS, colour = "black"),
      legend.title = element_text(size = FS_LEGEND),
      legend.text  = element_text(size = FS_LEGEND),
      legend.key.size = unit(0.5, "cm"),
      strip.text  = element_text(size = FS_LEGEND, face = "bold"),
      panel.grid  = element_blank(),
      plot.margin = margin(14, 14, 14, 14)
    )
}

# save a plot to svg / pdf / png / tiff at a given aspect (w:h) and unit size
save_pub <- function(plot, outdir, name, w_in = 10, h_in = 5, dpi = 600) {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  svglite::svglite(file.path(outdir, paste0(name, ".svg")),
                   width = w_in, height = h_in)
  print(plot); dev.off()
  grDevices::cairo_pdf(file.path(outdir, paste0(name, ".pdf")),
                       width = w_in, height = h_in, family = "Arial")
  print(plot); dev.off()
  ragg::agg_png(file.path(outdir, paste0(name, ".png")),
                width = w_in, height = h_in, units = "in", res = dpi)
  print(plot); dev.off()
  ragg::agg_tiff(file.path(outdir, paste0(name, ".tiff")),
                 width = w_in, height = h_in, units = "in", res = dpi)
  print(plot); dev.off()
  message("saved: ", name)
}

# project root + paths
ROOT <- "D:/AD-Multi-Omics-Integration"
PROC <- function(fig, f) file.path(ROOT, "figures", "20260525", fig,
                                   paste0(f, "_processed.csv"))
