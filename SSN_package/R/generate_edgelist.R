#' Generate Edge List from Microbial Data
#'
#' This function reads microbial count data, performs centered log-ratio transformation,
#' computes SparCC correlations, performs bootstrapping to calculate p-values,
#' and generates an edge list of significant correlations.
#'
#' @param filename The path to the input file containing microbial count data.
#' @param bootstraps The number of bootstrap samples to use (default is 1000).
#' @param p_value_threshold The significance threshold for p-values (default is 0.05).
#' @return A data frame containing the edge list with significant correlations.
#' @export
Generate_edgelist <- function(filename, bootstraps = 1000, p_value_threshold = 0.05) {
  # Load necessary libraries
  library(matrixStats)
  library(reshape2)
  library(SpiecEasi)
  library(data.table)

  # Define the centered log-ratio transformation function
  centeredLogRatio <- function(otu_table) {
    noZeros <- otu_table
    noZeros[noZeros == 0] <- NA
    geomeans <- apply(noZeros, 1, function(x) exp(mean(log(x), na.rm = TRUE)))
    clr_table <- log(noZeros / geomeans)
    clr_table[is.na(clr_table)] <- 0
    return(clr_table)
  }

  # Define the permutation function with replacement
  permute_w_replacement <- function(frame) {
    s <- nrow(frame)
    perm <- apply(frame, 2, function(x) sample(x, size = s, replace = TRUE))
    perm <- as.data.frame(perm)
    return(perm)
  }

  # Define the bootstrap correlations function
  bootstrap_correlations <- function(df, cor, samplename, bootstraps) {
    abs_cor <- abs(cor)
    n_sig <- matrix(0, nrow = nrow(abs_cor), ncol = ncol(abs_cor))
    rownames(n_sig) <- rownames(cor)
    colnames(n_sig) <- colnames(cor)

    for (i in 1:bootstraps) {
      bootstrap <- permute_w_replacement(df)
      perm_cor <- sampling_sparcc(bootstrap, samplename)
      n_sig[abs(perm_cor) >= abs_cor] <- n_sig[abs(perm_cor) >= abs_cor] + 1
    }

    p_vals <- n_sig / bootstraps
    diag(p_vals) <- 1
    p_vals <- as.data.frame(p_vals)
    rownames(p_vals) <- rownames(cor)
    colnames(p_vals) <- colnames(cor)
    return(p_vals)
  }

  # Define the sampling SparCC function
  sampling_sparcc <- function(frame_ori, samplename) {
    frame <- frame_ori[rownames(frame_ori) != samplename, ]
    x <- as.matrix(frame)
    sampleSize <- nrow(x)
    nodeD <- ncol(x)

    vmean <- colMeans(frame, na.rm = TRUE)
    x_dif <- array(0, dim = c(sampleSize, nodeD, nodeD))
    xT_dif <- array(0, dim = c(sampleSize, nodeD, nodeD))

    for (i in 1:sampleSize) {
      x_dif[i, , ] <- sweep(matrix(x[i, ], nrow = nodeD, ncol = nodeD, byrow = TRUE), 2, vmean)
      xT_dif[i, , ] <- t(x_dif[i, , ])
    }

    sumweight <- apply(x_dif * xT_dif, c(2, 3), sum, na.rm = TRUE)
    varx1 <- sqrt(colSums(x_dif^2, dims = 1)) %*% t(sqrt(colSums(xT_dif^2, dims = 1)))
    cor_content <- sumweight / varx1
    cor_content[is.na(cor_content)] <- 0
    cor_frame <- as.data.frame(cor_content)
    rownames(cor_frame) <- colnames(frame)
    colnames(cor_frame) <- colnames(frame)
    return(cor_frame)
  }

  # Read the data
  comb_table_ori <- fread(filename, data.table = FALSE)
  rownames(comb_table_ori) <- comb_table_ori[, 1]
  comb_table_ori <- comb_table_ori[, -1]

  # Normalize the data
  comb_table_ori <- t(t(comb_table_ori) / colSums(comb_table_ori))
  comb_table_ori[is.na(comb_table_ori)] <- 0

  # Apply centered log-ratio transformation
  comb_table_clr <- t(centeredLogRatio(comb_table_ori))

  # Compute Pearson correlation matrix
  comb_table_clrmatr <- cor(comb_table_clr, method = 'pearson', use = 'pairwise.complete.obs')
  comb_table_clrmatr[is.na(comb_table_clrmatr)] <- 0

  # Extract sample name from filename
  sample_name <- strsplit(basename(filename), '_')[[1]][3]

  # Compute SparCC correlations
  Framecor <- sampling_sparcc(comb_table_clr, sample_name)

  # Perform bootstrap to compute p-values
  perm_pval <- bootstrap_correlations(
    df = comb_table_clr,
    cor = Framecor,
    samplename = sample_name,
    bootstraps = bootstraps
  )

  # Filter correlations with p-value <= threshold
  sparcc_corfilter <- Framecor
  sparcc_corfilter[perm_pval > p_value_threshold] <- NA

  # Generate the edge list
  sparcc_corfilter_melt <- melt(as.matrix(sparcc_corfilter), na.rm = TRUE)
  colnames(sparcc_corfilter_melt) <- c('Node1', 'Node2', 'Correlation')

  # Remove self-loops if any
  sparcc_corfilter_melt <- sparcc_corfilter_melt[sparcc_corfilter_melt$Node1 != sparcc_corfilter_melt$Node2, ]

  # Remove duplicate edges (since the correlation matrix is symmetrical)
  sparcc_corfilter_melt <- sparcc_corfilter_melt[!duplicated(t(apply(sparcc_corfilter_melt[, 1:2], 1, sort))), ]

  # Return the edge list
  return(sparcc_corfilter_melt)
}
