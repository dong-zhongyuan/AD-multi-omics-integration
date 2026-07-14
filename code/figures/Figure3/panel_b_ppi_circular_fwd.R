# panel_b_ppi_circular_fwd.R — circular PPI network, FORWARD, via circlize chordDiagram.
# KO sectors in vermilion with big gene-name badges; target sectors gray, only shared labeled.
suppressPackageStartupMessages({ library(circlize) })

here <- os.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/src")
DATA <- file.path(dirname(here), "data")
OUT  <- file.path(dirname(here), "output")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

LETTER <- "b"
df <- read.csv(file.path(DATA, "fig3b_ppi_fwd_edges.csv"))
mat <- xtabs(effect_score ~ KO_gene + target, data = df)
shared <- unique(df$target[df$shared])
KO_COLOR <- "#D55E00"

render <- function(open_fn, close_fn) {
  circos.clear()
  circos.par(gap.degree = 4, start.degree = 90,
             cell.padding = c(0.02, 0, 0.02, 0),
             canvas.xlim = c(-1.3, 1.3), canvas.ylim = c(-1.3, 1.3))
  grid_col <- character(ncol(mat)); names(grid_col) <- colnames(mat)
  grid_col[] <- ifelse(colnames(mat) %in% shared, "#444444", "#BBBBBB")
  all_col <- c(setNames(rep(KO_COLOR, nrow(mat)), rownames(mat)), grid_col)
  chordDiagram(mat, grid.col = all_col,
               annotationTrack = "grid",
               preAllocateTracks = list(track.height = 0.18),
               transparency = 0.25)
  # KO labels (large, BLACK so they read at any zoom)
  for (sn in rownames(mat)) {
    xl <- get.cell.meta.data("xlim", sector.index = sn, track.index = 1)
    yl <- get.cell.meta.data("ylim", sector.index = sn, track.index = 1)
    circos.text(mean(xl), yl[2] + mm_y(0.8), sn, sector.index = sn, track.index = 1,
                facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),
                col = "#000000", cex = 1.4, font = 2)
  }
  # Shared-target labels
  for (sn in colnames(mat)) {
    if (!(sn %in% shared)) next
    xl <- get.cell.meta.data("xlim", sector.index = sn, track.index = 1)
    yl <- get.cell.meta.data("ylim", sector.index = sn, track.index = 1)
    circos.text(mean(xl), yl[2] + mm_y(0.6), sn, sector.index = sn, track.index = 1,
                facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),
                col = "#000000", cex = 1.2, font = 2)
  }
  circos.clear()
}

# PNG — larger canvas + margin so labels are not truncated
png(file.path(OUT, paste0(LETTER, ".png")), width = 2400, height = 2400, res = 300, bg = "white")
par(mar = c(1, 1, 1, 1))
render()
dev.off()
# PDF
pdf(file.path(OUT, paste0(LETTER, ".pdf")), width = 8, height = 8)
par(mar = c(1, 1, 1, 1))
render()
dev.off()
cat("Saved", LETTER, "png/pdf\n")
