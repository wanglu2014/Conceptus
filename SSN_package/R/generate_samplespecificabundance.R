#' Generate Sample-Specific Abundance Data
#'
#' This function generates sample-specific abundance data by performing leave-one-out cross-validation (LOOCV) on phyloseq objects. It subsets the data based on specified parameters, fills missing values, and writes the output to files.
#'
#' @param mypath A character string specifying the path to the directory containing the data files.
#' @param phylo_name A character string specifying the name of the phyloseq RData file to load.
#' @param nnum An integer specifying the number of top taxa to select.
#' @param thresholdi A numeric value between 0 and 1 specifying the threshold for missing data.
#' @return No return value. The function writes output files to the specified directory.
#' @export
generate_samplespecificabundance <- function(mypath, phylo_name, nnum, thresholdi) {
  # Load necessary libraries
  library(phyloseq)
  library(stringr)
  library(fantaxtic)
  library(microbiome)
  library(data.table)
  library(dplyr)
  library(tibble)

  # Ensure the inputs are correct
  if (!dir.exists(mypath)) {
    stop("The specified path does not exist.")
  }

  # Load the phyloseq object
  load(file.path(mypath, phylo_name))

  # Extract the phylotxt from the filename
  phylotxt <- str_split_fixed(phylo_name, '_', 3)[, 2]

  # Get the metadata
  meta <- sample_data(ps.ng.tax)

  samp_dat <- sam_data(ps.ng.tax)
  rownames(samp_dat) <- sample_names(ps.ng.tax)
  # Loop over each group in the metadata
  for (groupgreat in unique(samp_dat$type)) {
    # Subset the phyloseq object to the current group
    #phylo_Glom <- subset_samples(ps.ng.tax, type == groupgreat)
    samples_to_keep <- rownames(samp_dat)[samp_dat$type == groupgreat]
    phylo_Glom <- prune_samples(samples_to_keep, ps.ng.tax)

    # Get the top nnum taxa
    phylo_Glom <- fantaxtic::get_top_taxa(phylo_Glom, n = nnum, relative = TRUE, discard_other = TRUE)

    # Transform to compositional data
    phylo_Glom <- microbiome::transform(phylo_Glom, transform = 'compositional')

    # Get the OTU table
    otutable <- otu_table(phylo_Glom)
    otutable[otutable == 0] <- NA
    otutableT <- as.data.frame(t(otutable))

    # Prepare the data
    SSN20 <- otutableT
    subject_frame_num <- SSN20
    subject_frame_num[subject_frame_num == 0] <- NA
    subject_frame_num <- subject_frame_num[, which(colMeans(is.na(subject_frame_num)) < thresholdi)]
    SSN20 <- subject_frame_num

    # Generate combinations for LOOCV
    combname19 <- combn(rownames(SSN20), nrow(SSN20) - 1)

    apply(combname19, 2, function(samples19) {
      samplename <- setdiff(rownames(SSN20), samples19)
      SSN19 <- SSN20[samples19, ]
      SSN1 <- SSN20[samplename, , drop = FALSE]

      # Impute mean for missing values
      for (i in 1:ncol(SSN19)) {
        colmeani <- mean(SSN19[, i], na.rm = TRUE)
        SSN19[is.na(SSN19[, i]), i] <- colmeani
        SSN1[, i][is.na(SSN1[, i])] <- colmeani
      }

      # Combine SSN19 and SSN1
      SSN19add1 <- data.table::rbindlist(list(SSN19, SSN1), fill = TRUE)
      rownames(SSN19add1) <- c(rownames(SSN19), samplename)

      # Prepare the data frame
      SSN19add1dif <- SSN19add1 %>%
        rownames_to_column(var = 'Row.names') %>%
        column_to_rownames(var = 'Row.names')

      # Write the output file
      output_filename <- paste0(mypath, phylotxt, "_", groupgreat, "_", samplename, "_", nnum, "_", thresholdi, "_singletypein.txt")
      write.table(t(SSN19add1dif), file = output_filename, row.names = TRUE, sep = "\t")
    })
  }
}
