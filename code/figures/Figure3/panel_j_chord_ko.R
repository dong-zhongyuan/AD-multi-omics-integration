# panel_j_chord_ko.R — KO×KO chord diagram (shared perturbation targets).
# Two KOs are linked if they both perturb the same target gene; edge weight = # shared targets.
# Reverse direction only (has the richest shared-target structure: 14 hubs, including
# GNG11 hit by 3 KOs forming a triangle).
suppressPackageStartupMessages({ library(circlize); library(dplyr) })

here <- os.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/src")
DATA <- file.path(dirname(here), "data")
OUT  <- file.path(dirname(here), "output")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

LETTER <- "j"
df <- read.csv(file.path(DATA, "fig3c_ppi_rev_edges.csv"))
kos <- sort(unique(df$KO_gene))

# Build KO x target incidence (1 if KO hits target), then KO x KO co-occurrence = t(X) %*% X style.
inc <- xtabs(~ KO_gene + target, data = df)              # rows = KO, cols = target, values = 1
mat <- tcrossprod(inc)                                    # KO x KO, diagonal = #targets per KO
diag(mat) <- 0                                            # no self-loops
mat <- mat[kos, kos]

KO_COLOR <- "#D55E00"

render_body <- function() {
  circos.clear()
  circos.par(gap.degree = 3, start.degree = 90,
             cell.padding = c(0.02, 0, 0.02, 0),
             canvas.xlim = c(-1.35, 1.35), canvas.ylim = c(-1.35, 1.35))
  grid_col <- setNames(rep(KO_COLOR, nrow(mat)), rownames(mat))
  chordDiagram(mat, grid.col = grid_col,
               symmetric = TRUE, transparency = 0.2,
               annotationTrack = "grid",
               preAllocateTracks = list(track.height = 0.16))
  for (sn in rownames(mat)) {
    xl <- get.cell.meta.data("xlim", sector.index = sn, track.index = 1)
    yl <- get.cell.meta.data("ylim", sector.index = sn, track.index = 1)
    circos.text(mean(xl), yl[2] + mm_y(0.4), sn, sector.index = sn, track.index = 1,
                facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),
                col = "#000000", cex = 1.05, font = 2)
  }
  circos.clear()
}

# larger canvas + margin so labels are not truncated
png(file.path(OUT, paste0(LETTER, ".png")), width = 2400, height = 2400, res = 300, bg = "white")
par(mar = c(1, 1, 1, 1)); render_body()
dev.off()
pdf(file.path(OUT, paste0(LETTER, ".pdf")), width = 8, height = 8)
par(mar = c(1, 1, 1, 1)); render_body()
dev.off()
cat("Saved", LETTER, "png/pdf\n")
