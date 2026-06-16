#' Generate SSN
#'
#' This function generates SSN from the provided data.
#'
#' @param mypath The path to the directory containing the data files.
#' @param phylo_list A list of .Rdata files to be processed.
#' @export
generate_SSN <-
function(mypath = "E:/diabetes0409/", phylo_list = c('aphy_diabetesmetross_resp.Rdata')) {
  for (phylo_name in phylo_list) {
    load(paste0(mypath, phylo_name))

    phylotxt <- tools::file_path_sans_ext(basename(phylo_name))

    parts <- strsplit(phylotxt, "_")[[1]]

    phylotxt <- parts[2]

    filelist <- list.files(pattern = paste0('^', phylotxt, '.*txt_comselP.csv$'), path = mypath)
    SSN_list <- list()

    for (file19 in filelist) {
      filesplit <- strsplit(file19, "_")[[1]]
      G1 <- read.csv(paste0(mypath, file19))
      G1$filename <- file19
      G1$edges_OTU <- paste0(G1$index, ';.;', G1$variable)
      SSN19add1 <- acast(G1, samples ~ edges_OTU, value.var = 'value')

      group <- filesplit[2]
      samplename <- filesplit[3]
      topn <- filesplit[4]
      thre <- filesplit[5]

      if (samplename %in% rownames(SSN19add1)) {
        SSN19add1 <- as.data.frame(SSN19add1)
        edgelist <- data.frame(SSN19add1[samplename, ], row.names = samplename, check.names = FALSE)
        colnames(edgelist) <- colnames(SSN19add1)
        SSN_list[[file19]] <- edgelist
        edgelist <- as.data.frame(t(edgelist), check.names = FALSE) %>%
          rownames_to_column(var = 'edges_OTU') %>%
          separate(edges_OTU, into = c('V1', 'V2'), sep = ';.;')
        edgelist <- edgelist[!is.na(edgelist[samplename]), ]
        g <- graph_from_edgelist(as.matrix(edgelist[, c('V1', 'V2')]), directed = FALSE)
        E(g)$weight <- unlist(edgelist[samplename])
        E(g)$weight <- format(E(g)$weight, digits = 3)
        write_graph(g, file = paste0(mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'), format = "gml")
      } else {
        print(paste0('No file ', mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'))
        G1 <- make_empty_graph(n = 0, directed = FALSE)
        write_graph(G1, file = paste0(mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'), format = "gml")
      }
    }

    SSNall <- rbindlist(SSN_list, fill = TRUE, use.names = TRUE, idcol = 'samplename') %>%
      melt(na.rm = TRUE)
    write.csv(SSNall, paste0(mypath, phylotxt, "_comselSSN.csv"), row.names = FALSE)

    meta <- meta(ps.ng.tax)
    meta2 <- meta
    row.names(meta2) <- gsub('X', '', row.names(meta2))
    meta <- rbind(meta, meta2)

    attrlist <- list()
    for (file in list.files(path = mypath, pattern = paste0(phylotxt, '.*.gml$'))) {
      G1 <- read_graph(paste0(mypath, file), format = "gml")
      tryCatch({
        E(G1)$weight <- abs(as.numeric(as.character(E(G1)$weight)))
        G_frame <- data.frame(
          mean_degree = mean(degree(G1)),
          average_path = average.path.length(G1),
          diameter = diameter(G1, unconnected = TRUE),
          mean_betweenness = mean(betweenness(G1)),
          mean_closeness = mean(closeness(G1)),
          edgenumber = length(E(G1)),
          nodenumber = length(V(G1))
        )
        rownames(G_frame) <- file
        attrlist[[file]] <- G_frame
      }, error = function(error_message) {
        message("Error is")
        print(file)
        attrlist[[file]] <- data.frame(mean_degree = 0, average_path = 0, diameter = 0, mean_betweenness = 0, mean_closeness = 0, edgenumber = 0, nodenumber = 0)
        return(NA)
      })
    }

    attrtable <- bind_rows(attrlist, .id = 'gml')
    attrtable$samples <- str_split_fixed(rownames(attrtable), '_', 5)[, 3]
    attrtable <- attrtable %>%
      merge(meta, by.x = 'samples', by.y = 'row.names')
    write.csv(attrtable, paste0(mypath, phylotxt, "_type_attrtable_metaSSN.csv"), row.names = TRUE)
  }
}
