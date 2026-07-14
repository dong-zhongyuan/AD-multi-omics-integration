# panel_d_radial_genki.R — radial bar chart, 4 sectors of EXACTLY 90deg each.
# 32 bars total (4 groups x 8 KOs), no inter-group gaps -> coord_polar gives each
# group exactly 90deg. No outer ring (removed per request). Sector identity (omics +
# role) shown via a discrete legend.
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
})

here <- os.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/src")
DATA <- file.path(dirname(here), "data")
OUT  <- file.path(dirname(here), "output")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

N_TOP <- 8
prot  <- read.csv(file.path(DATA, "fig3d_genki_proteomics.csv"))
trans <- read.csv(file.path(DATA, "fig3d_genki_transcriptomics.csv"))

prep <- function(df) {
  do.call(rbind, lapply(split(df, df$direction), function(s) {
    s[order(-s$kl), ][seq_len(min(N_TOP, nrow(s))), ]
  }))
}
prot$omics <- "Proteomics"; trans$omics <- "Transcriptomics"
big <- rbind(prep(prot), prep(trans))
big$kl_log <- log10(pmax(big$kl, 1e-2))
big$group  <- factor(paste(big$omics, substr(big$direction,1,1), sep="-"),
                     levels = c("Proteomics-F","Proteomics-R",
                                "Transcriptomics-F","Transcriptomics-R"))
big <- big[order(big$group, -big$kl_log), ]
# 32 bars, x = 1..32, no gaps -> each group spans exactly 8/32 = 90deg.
big$x <- seq_len(nrow(big))

PAL <- c("Proteomics-F"="#D55E00", "Proteomics-R"="#E69F00",
         "Transcriptomics-F"="#0072B2", "Transcriptomics-R"="#56B4E9")
ROLE <- c("Proteomics-F"="Proteomics / Diagnosis",
          "Proteomics-R"="Proteomics / Therapy",
          "Transcriptomics-F"="Transcriptomics / Diagnosis",
          "Transcriptomics-R"="Transcriptomics / Therapy")

YMAX <- max(big$kl_log)
big$sector <- factor(ROLE[as.character(big$group)], levels = ROLE)

p <- ggplot(big) +
  geom_col(aes(x = x, y = kl_log, fill = sector), width = 0.9,
           color = "white", linewidth = 0.3) +
  geom_text(aes(x = x, y = kl_log + YMAX * 0.04, label = KO_gene),
            size = 3.0, color = "grey20", fontface = "bold") +
  scale_fill_manual(values = setNames(PAL, ROLE), name = NULL) +
  scale_y_continuous(limits = c(-YMAX * 0.45, YMAX * 1.10)) +
  coord_polar(start = 0) +
  theme_minimal(base_family = "sans", base_size = 11) +
  theme(axis.title = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank(), panel.grid = element_blank(),
        plot.margin = margin(0, 0, 0, 0),
        legend.position = "right",
        legend.text = element_text(size = 9, face = "bold"),
        legend.key.size = unit(0.4, "cm"),
        legend.margin = margin(0, 0, 0, 0),
        legend.box.spacing = unit(0.05, "cm"))

for (ext in c("png","pdf"))
  ggsave(file.path(OUT, paste0("d.", ext)), p,
         width = 6.0, height = 5.6, dpi = 300, units = "in", bg = "white")
cat("Saved d.png/pdf\n")
