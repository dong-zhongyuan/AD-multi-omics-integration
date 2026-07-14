# panel_i_sankey_bubble.R — combined sankey + aligned bubble panel.
# Left: gene -> pathway sankey (ggsankeyfier). Right: bubble chart (x=-log10 adj p,
# y aligned to pathway's sankey y-position, size=gene count, color=hit ratio).
# Two facets stacked: Forward (top), Reverse (bottom). Replaces the former bar+bubble panels.
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(ggsankeyfier); library(patchwork)
})

here <- os.path.join(str(PROJECT_ROOT), "output/Figures_final/Figure3/src")
DATA <- file.path(dirname(here), "data")
OUT  <- file.path(dirname(here), "output")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

LETTER <- "i"
edges  <- read.csv(file.path(DATA, "fig_sankey_edges.csv"))
summary_df <- read.csv(file.path(DATA, "fig_bubble_summary.csv"))

# Shorten pathway labels (strip GO id)
short_term <- function(t) {
  t <- sub(" \\(GO:[0-9]+\\)$", "", as.character(t))
  ifelse(nchar(t) > 38, paste0(substr(t, 1, 36), ".."), t)
}
edges$pathway_short <- short_term(edges$pathway)
summary_df$pathway_short <- short_term(summary_df$pathway)

build_direction <- function(edges, summary_df, direction_label, fill_col) {
  e <- edges %>% filter(direction == direction_label)
  s <- summary_df %>% filter(direction == direction_label)
  if (nrow(e) == 0) return(NULL)
  e <- e %>% mutate(wt = 1)
  # long form for ggsankeyfier: stages_from = c(gene, pathway_short), values_from = wt
  long <- pivot_stages_longer(e,
                              stages_from = c("gene", "pathway_short"),
                              values_from = "wt")
  v_space <- 0.6
  pos <- position_sankey(v_space = v_space, order = "as_is", nudge_x = 0.06)
  # node colors: genes gray, pathways the direction color
  node_pal <- c(fill_col)
  genes_unique <- unique(e$gene)
  gene_cols <- setNames(rep("#BBBBBB", length(genes_unique)), genes_unique)
  path_cols <- setNames(rep(fill_col, length(unique(e$pathway_short))), unique(e$pathway_short))
  node_cols <- c(gene_cols, path_cols)

  sankey_p <- ggplot(long,
        aes(x = stage, y = wt, group = node,
            connector = connector, edge_id = edge_id, fill = node)) +
    geom_sankeyedge(position = pos, fill = "grey85", color = "grey70") +
    geom_sankeynode(position = pos, alpha = 0.9) +
    scale_x_discrete(expand = expansion(add = c(0.6, 0))) +
    scale_fill_manual(values = node_cols, guide = "none") +
    labs(x = NULL, y = NULL, title = direction_label) +
    theme_minimal(base_family = "sans", base_size = 13) +
    theme(legend.position = "none", panel.border = element_blank(),
          axis.text = element_blank(), axis.ticks = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0.5, size = 15))

  # Extract pathway y-coordinates from the built sankey node layer (layer 2).
  # Node layer has node_id + y/ymin/ymax + x (stage position). Map node_id -> name
  # by walking the long data in node_id order (ggsankeyfier assigns ids by appearance).
  gb <- ggplot_build(sankey_p)
  node_layer <- gb$data[[2]]
  # stage x positions (left stage = min x, right stage = max x)
  stages <- sort(unique(node_layer$x))
  # within each stage, nodes ordered top-to-bottom by descending y
  nl <- node_layer %>%
    arrange(x, desc(y)) %>%
    group_by(x) %>%
    summarise(nodes = list(node_id), ymins = list(ymin), ymaxs = list(ymax),
              ys = list((ymin + ymax)/2), .groups = "drop")
  # Build name lookup per stage from long data (unique nodes per stage, same order)
  stage_names <- long %>% distinct(stage, node) %>% group_by(stage) %>% group_split()
  # assemble lookup
  lookup <- data.frame()
  stage_x <- stages
  for (i in seq_along(stage_names)) {
    st_df <- stage_names[[i]]
    st_x <- stage_x[i]
    nl_row <- nl[nl$x == st_x, ]
    if (nrow(nl_row) == 0) next
    # match by order (both should be aligned); fall back to position
    n <- length(nl_row$nodes[[1]])
    nm <- st_df$node[seq_len(min(n, nrow(st_df)))]
    lookup <- rbind(lookup, data.frame(
      node_id = nl_row$nodes[[1]][seq_len(length(nm))],
      label = nm,
      y = nl_row$ys[[1]][seq_len(length(nm))],
      ymin = nl_row$ymins[[1]][seq_len(length(nm))],
      ymax = nl_row$ymaxs[[1]][seq_len(length(nm))]))
  }
  y_pos <- lookup %>% filter(label %in% s$pathway_short)

  bub <- s %>% left_join(y_pos, by = c("pathway_short" = "label"))
  bub$nlp <- -log10(bub$adj_p)
  y_all_range <- if (nrow(y_pos)) range(c(lookup$ymin, lookup$ymax), na.rm = TRUE) else c(0,1)
  bubble_p <- ggplot(bub, aes(x = nlp, y = y, size = n_genes_overlap, color = hit_ratio)) +
    geom_point(alpha = 0.85) +
    scale_color_gradient(low = "#FEE08B", high = fill_col, name = "Hit ratio") +
    scale_size_continuous(name = "Gene count", range = c(3, 9)) +
    scale_y_continuous(limits = y_all_range) +
    labs(x = expression(-log[10](adj~p)), y = NULL) +
    theme_minimal(base_family = "sans", base_size = 13) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          axis.text.x = element_text(size = 11),
          axis.title.x = element_text(size = 13, face = "bold"),
          legend.position = "right",
          legend.text = element_text(size = 11),
          legend.title = element_text(size = 12, face = "bold"),
          plot.title = element_text(face = "bold", hjust = 0.5, size = 15))

  sankey_p + bubble_p + plot_layout(widths = c(3, 1.2), guides = "collect")
}

p_fwd <- build_direction(edges, summary_df, "Forward",  "#0072B2")
p_rev <- build_direction(edges, summary_df, "Reverse",  "#D55E00")

# stack vertically
combo <- (p_fwd / p_rev) + plot_layout(heights = c(1, 1))

for (ext in c("png","pdf"))
  ggsave(file.path(OUT, paste0(LETTER, ".", ext)), combo,
         width = 12, height = 9, dpi = 300, units = "in", bg = "white",
         limitsize = FALSE)
cat("Saved", LETTER, "png/pdf\n")
