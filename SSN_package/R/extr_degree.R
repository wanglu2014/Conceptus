#' Extract Degree and Merge with Metadata and OTU Table
#'
#' This function reads .gml graph files from a specified directory, computes node degrees, extracts graph attributes, merges them with metadata and OTU tables from corresponding phyloseq objects, and writes the results to CSV files.
#'
#' @param mypath A character string specifying the directory containing the .gml files.
#' @param rdata_path A character string specifying the directory containing the .Rdata files (phyloseq objects).
#' @param output_path A character string specifying the directory where output CSV files will be saved.
#' @return No return value. The function writes output CSV files to the specified output directory.
#' @export
extr_degree <- function(mypath, rdata_path, output_path) {
  # Load necessary libraries
  library(igraph)
  library(data.table)
  library(dplyr)
  library(stringr)
  library(reshape2)
  library(tibble)
  library(microbiome)

  graphattr_list <- list()

  # Read and process .gml files
  gml_files <- list.files(path = mypath, pattern = ".*\\.gml$", full.names = TRUE)

  for (file in gml_files) {
    G1 <- read_graph(file, format = c("gml"))
    V(G1)$degree <- degree(G1)

    attr_graph <- lapply(list.vertex.attributes(G1), function(x) {
      as.data.frame(t(get.vertex.attribute(G1, x)))
    })
    attr_table <- t(rbindlist(attr_graph, fill = TRUE))
    colnames(attr_table) <- list.vertex.attributes(G1)
    graphattr_list[[basename(file)]] <- as.data.frame(attr_table)
  }

  # Combine graph attributes into a single data frame
  graphlist_table <- bind_rows(graphattr_list, .id = "filename")

  # Extract group, phylotxt, and samples from filename
  graphlist_table <- graphlist_table %>%
    mutate(
      group = str_extract(filename, "(?<=_)[^_]+_[^_]+(?=_\\d{4}\\.gml$)"),
      phylotxt = str_split_fixed(filename, "_", 3)[, 1],
      samples = str_split_fixed(filename, "_", 4)[, 3]
    )

  # Loop over unique groups and phylotypes
  for (group_name in unique(graphlist_table$group)) {
    group_data <- graphlist_table %>% filter(group == group_name)

    for (phylotxtname in unique(group_data$phylotxt)) {
      phylo_mtypedegree <- group_data %>% filter(phylotxt == phylotxtname)
      rdata_file <- file.path(rdata_path, paste0('aphy_', phylotxtname, '_resp.Rdata'))

      if (file.exists(rdata_file)) {
        load(rdata_file)
        meta_data <- meta(ps.ng.tax)
        otu_data <- as.data.frame(t(otu_table(ps.ng.tax))) %>%
          rownames_to_column(var = 'Row.names')
        meta_data <- meta_data %>%
          rownames_to_column(var = 'Row.names')

        # Merge OTU data with metadata
        otu_meta <- left_join(otu_data, meta_data, by = 'Row.names')

        # Create degree table
        degreetable <- acast(
          data = phylo_mtypedegree,
          formula = samples ~ name,
          value.var = 'degree',
          fun.aggregate = mean
        )
        degreetable <- as.data.frame(degreetable)

        # Handle missing samples
        all_samples <- meta_data$Row.names
        missing_samples <- setdiff(all_samples, rownames(degreetable))

        if (length(missing_samples) > 0) {
          missing_df <- data.frame(matrix(0, nrow = length(missing_samples), ncol = ncol(degreetable)))
          rownames(missing_df) <- missing_samples
          colnames(missing_df) <- colnames(degreetable)
          degreetable <- rbind(degreetable, missing_df)
        }

        # Merge degree table with metadata
        degreemeta_table <- degreetable %>%
          rownames_to_column(var = 'Row.names') %>%
          left_join(meta_data, by = 'Row.names') %>%
          column_to_rownames(var = 'Row.names')

        # Check for duplicates
        duplicates <- phylo_mtypedegree %>%
          group_by(samples, name) %>%
          filter(n() > 1) %>%
          ungroup()

        # Print duplicates if any
        if (nrow(duplicates) > 0) {
          print(paste("Duplicates found in phylotype:", phylotxtname, "group:", group_name))
          print(duplicates)
        }

        # Merge degree-meta table with OTU data
        final_table <- degreemeta_table %>%
          rownames_to_column(var = 'Row.names') %>%
          left_join(otu_data, by = 'Row.names') %>%
          column_to_rownames(var = 'Row.names')

        # Write the output CSV file
        output_filename <- paste0(phylotxtname, '_', group_name, '_metaOTUdegree.csv')
        write.csv(final_table, file = file.path(output_path, output_filename), row.names = TRUE)

      } else {
        warning(paste("Rdata file not found:", rdata_file))
      }
    }
  }
}
