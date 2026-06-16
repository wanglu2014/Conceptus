#' Create Phyloseq Object from CSV Files
#'
#' This function reads in abundance and metadata CSV files, creates a phyloseq object named ps.ng.tax, and saves it as an Rdata file.
#'
#' @param abundance_file The path to the abundance CSV file.
#' @param metadata_file The path to the metadata CSV file.
#' @param output_file The name of the output Rdata file (e.g., 'aphy_sample.Rdata').
#' @return The phyloseq object ps.ng.tax.
#' @export
create_phyloseq_object <- function(abundance_file, metadata_file, output_file) {
  # Load necessary libraries
  library(phyloseq)

  # Read in the abundance data
  abundance_data <- read.csv(abundance_file, row.names = 1, check.names = FALSE)

  # Convert abundance data to matrix
  otu_matrix <- as.matrix(abundance_data)

  # Create OTU Table
  otu_table_ps <- otu_table(otu_matrix, taxa_are_rows = TRUE)

  # Read in the metadata
  metadata <- read.csv(metadata_file, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)

  # Create Sample Data
  sample_data_ps <- sample_data(metadata)

  # Create the phyloseq object
  ps.ng.tax <- phyloseq(otu_table_ps, sample_data_ps)

  # Save the phyloseq object
  save(ps.ng.tax, file = output_file)

  # Return the phyloseq object
  return(ps.ng.tax)
}

