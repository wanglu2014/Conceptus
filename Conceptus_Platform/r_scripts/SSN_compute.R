#!/usr/bin/env Rscript
# SSN_compute.R - Subject-Specific Network Computation
# Original script content with logging enabled

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 5) {
  cat("Usage: Rscript SSN_compute.R <otu_file> <meta_file> <threshold> <seed> <output_dir>\n")
  quit(status = 1)
}

otu_file <- args[1]
meta_file <- args[2]
threshold <- as.numeric(args[3])
seed <- as.numeric(args[4])
output_dir <- args[5]

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Enable logging to file in output directory
log_file <- file.path(output_dir, "ssn_debug.log")
con <- file(log_file, open = "wt")
sink(con, split = TRUE) 
# split=TRUE not supported for message connection on some systems
sink(con, type = "message")
cat(sprintf("Logging started at %s\n", Sys.time()))

suppressPackageStartupMessages({
  library(readr)
  library(tidyverse)
  library(Matrix)
  library(reshape2)
  library(SpiecEasi)
  library(phyloseq)
  library(igraph)
})


otu_file <- args[1]
meta_file <- args[2]
threshold <- as.numeric(args[3])
seed <- as.numeric(args[4])
output_dir <- args[5]

cat("=== SSN Computation Started ===\n")
cat(paste("OTU file:", otu_file, "\n"))
cat(paste("Meta file:", meta_file, "\n"))
cat(paste("Threshold:", threshold, "\n"))
cat(paste("Seed:", seed, "\n"))
cat(paste("Output dir:", output_dir, "\n"))

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
setwd(output_dir)

# Read input files
cat("Reading input files...\n")
meta <- read_csv(meta_file, show_col_types = FALSE)
otu_raw <- read_csv(otu_file, show_col_types = FALSE)

# Get sample ID column name from OTU file (first column)
sample_col <- colnames(otu_raw)[1]
cat(paste("OTU Sample column:", sample_col, "\n"))

# Get sample ID column name from Meta file (first column)
meta_sample_col <- colnames(meta)[1]
cat(paste("Meta Sample column:", meta_sample_col, "\n"))

# Get Group column name from Meta file (assume second column)
# Or look for common names like "Group", "Subgroup", "condition"
if ("Subgroup" %in% colnames(meta)) {
  group_col <- "Subgroup"
} else if (ncol(meta) >= 2) {
  group_col <- colnames(meta)[2]
} else {
  stop("Metadata file must have at least 2 columns (Sample ID and Group)")
}
cat(paste("Group column:", group_col, "\n"))

# Merge OTU with metadata
# Use all_of() to handle column names safely
OTUcount <- otu_raw %>%
  pivot_longer(cols = -all_of(sample_col), names_to = "OTU", values_to = "count") %>%
  merge(meta, by.x = sample_col, by.y = meta_sample_col)

# Ensure consistent group column name for downstream code
if (group_col != "Subgroup") {
  OTUcount$Subgroup <- OTUcount[[group_col]]
}

cat(paste("Total samples:", length(unique(OTUcount[[sample_col]])), "\n"))

# Get unique subgroups
subgroups <- unique(OTUcount$Subgroup)
cat(paste("Subgroups found:", paste(subgroups, collapse = ", "), "\n"))

# Set igraph arpack parameters
# Commented out as it causes 'object of type closure is not subsettable' in newer igraph versions
# arpack_defaults$maxiter <- 9999999

# Network attribute extraction function
extract_network_attrs <- function(g) {
  tryCatch({
    E(g)$weight <- abs(E(g)$weight)
    data.frame(
      mean_degree = mean(degree(g)),
      edge_number = length(E(g)),
      node_number = length(V(g))
    )
  }, error = function(e) {
    data.frame(mean_degree = NA, edge_number = NA, node_number = NA)
  })
}

# Results storage
all_attrs <- list()

# Process each subgroup
for (grp in subgroups) {
  cat(paste("\nProcessing subgroup:", grp, "\n"))

  grp_data <- OTUcount %>% filter(Subgroup == grp)
  samples <- unique(grp_data[[sample_col]])
  cat(paste("  Samples in group:", length(samples), "\n"))

  # Create OTU matrix for the group
  otu_matrix <- grp_data %>%
    select(all_of(sample_col), OTU, count) %>%
    pivot_wider(names_from = OTU, values_from = count, values_fill = 0) %>%
    column_to_rownames(sample_col) %>%
    as.matrix()

  # Filter OTUs based on threshold
  otu_matrix[otu_matrix == 0] <- NA
  keep_cols <- colMeans(is.na(otu_matrix)) < threshold
  otu_matrix <- otu_matrix[, keep_cols, drop = FALSE]
  otu_matrix[is.na(otu_matrix)] <- 0

  cat(paste("  OTUs after filtering:", ncol(otu_matrix), "\n"))

  # Skip if too few OTUs
  if (ncol(otu_matrix) < 3) {
    cat("  Skipping: too few OTUs\n")
    next
  }

  # Build group-level network
  cat("  Building group network...\n")
  tryCatch({
    phylo_obj <- phyloseq(otu_table(otu_matrix, taxa_are_rows = FALSE))
    spiec_out <- spiec.easi(
      phylo_obj,
      method = 'mb',
      lambda.min.ratio = 1e-2,
      nlambda = 20,
      pulsar.params = list(rep.num = 50, seed = seed)
    )

    # Save group network
    g_group <- adj2igraph(getRefit(spiec_out))
    gml_file <- paste0(grp, "_network.gml")
    write_graph(g_group, file = gml_file, format = "gml")

    # Extract group network attributes
    grp_attrs <- extract_network_attrs(g_group)
    grp_attrs$sample <- paste0(grp, "_group")
    grp_attrs$subgroup <- grp
    all_attrs[[paste0(grp, "_group")]] <- grp_attrs

    cat(paste("  Group network: edges=", grp_attrs$edge_number,
              ", nodes=", grp_attrs$node_number, "\n"))

  }, error = function(e) {
    cat(paste("  Error building group network:", e$message, "\n"))
  })

  # Build individual leave-one-out networks
  cat("  Building individual networks (leave-one-out)...\n")

  for (i in seq_along(samples)) {
    samp <- samples[i]

    # Leave-one-out: use all samples except current one
    other_samples <- setdiff(samples, samp)

    if (length(other_samples) < 3) {
      cat(paste("    Skipping", samp, ": too few samples for LOO\n"))
      next
    }

    tryCatch({
      loo_matrix <- otu_matrix[other_samples, , drop = FALSE]

      # Re-filter for this subset
      loo_matrix[loo_matrix == 0] <- NA
      keep_cols_loo <- colMeans(is.na(loo_matrix)) < threshold
      if (sum(keep_cols_loo) < 3) next
      loo_matrix <- loo_matrix[, keep_cols_loo, drop = FALSE]
      loo_matrix[is.na(loo_matrix)] <- 0

      phylo_loo <- phyloseq(otu_table(loo_matrix, taxa_are_rows = FALSE))
      spiec_loo <- spiec.easi(
        phylo_loo,
        method = 'mb',
        lambda.min.ratio = 1e-2,
        nlambda = 15,
        pulsar.params = list(rep.num = 30, seed = seed)
      )

      g_loo <- adj2igraph(getRefit(spiec_loo))

      # Save individual network
      gml_file <- paste0(samp, "_network.gml")
      write_graph(g_loo, file = gml_file, format = "gml")

      # Extract attributes
      samp_attrs <- extract_network_attrs(g_loo)
      samp_attrs$sample <- samp
      samp_attrs$subgroup <- grp
      all_attrs[[samp]] <- samp_attrs

      if (i %% 5 == 0) {
        cat(paste("    Processed", i, "/", length(samples), "samples\n"))
      }

    }, error = function(e) {
      cat(paste("    Error for", samp, ":", e$message, "\n"))
    })
  }
}

# Combine all attributes
cat("\n=== Combining Results ===\n")
if (length(all_attrs) > 0) {
  attr_table <- bind_rows(all_attrs)

  # Reorder columns
  attr_table <- attr_table %>%
    select(sample, subgroup, mean_degree, edge_number, node_number) %>%
    arrange(sample)

  # Add outcome column (SS/SN = 1, CS/CN = 0)
  attr_table <- attr_table %>%
    mutate(outcome = case_when(
      grepl("^S", subgroup) & grepl("S$", subgroup) ~ 1,  # SS
      grepl("^S", subgroup) & grepl("N$", subgroup) ~ 1,  # SN
      TRUE ~ 0  # CS, CN
    ))

  # Save results
  output_file <- "attrtable_SSN.csv"
  write_csv(attr_table, output_file)
  cat(paste("Results saved to:", file.path(output_dir, output_file), "\n"))
  cat(paste("Total records:", nrow(attr_table), "\n"))

  # Also create Conceptus-compatible metrics file
  metrics_file <- "metrics.csv"
  metrics_table <- attr_table %>%
    filter(!grepl("_group$", sample)) %>%
    rename(
      closeness = mean_degree,
      diameter = edge_number,
      node_number = node_number,
      group = subgroup
    ) %>%
    select(sample, closeness, diameter, node_number, group, outcome)

  write_csv(metrics_table, metrics_file)
  cat(paste("Metrics file saved to:", file.path(output_dir, metrics_file), "\n"))

} else {
  cat("No results to save.\n")
  quit(status = 1)
}

cat("\n=== SSN Computation Completed ===\n")
