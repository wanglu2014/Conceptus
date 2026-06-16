#' Extract Network Attributes
#'
#' This function extracts network attributes from the provided graph.
#'
#' @param graph The input graph from which attributes are extracted.
#' @return A data frame containing the extracted attributes.
#' @export
extr_netattr <-
function(mypath = "E:/diabetes0409/", phylo_list = c('aphy_diabetesmetross_resp.Rdata')) {
  library(data.table)
  library(dplyr)
  library(igraph)
  library(reshape2)
  library(stringr)
  library(tidyr)

  for (phylo_name in phylo_list) {
    load(paste0(mypath, phylo_name))

    # 从文件名中提取 phylotxt
    phylotxt <- str_extract(phylo_name, "aphy_[^_]+")

    # 使用 strsplit() 按下划线分割字符串
    parts <- strsplit(phylotxt, "_")[[1]]

    # parts 是一个字符向量：c("aphy", "SRP135559", "resp", "with", "tax")

    # 提取第二个元素
    phylotxt <- parts[2]

    filelist <- list.files(pattern = paste0('^', phylotxt, '.*txt_comselP.csv$'), mypath)
    SSN_list <- list()

    for (file19 in filelist) {
      filesplit <- str_split_fixed(file19, pattern = '_', n = 7)
      G1 <- read.csv(paste0(mypath, file19))
      G1$filename <- file19
      G1$edges_OTU <- paste0(G1$index, ';.;', G1$variable)
      SSN19add1 <- data.frame(reshape2::acast(data = G1, formula = samples ~ edges_OTU, value.var = 'value'), check.names = FALSE)

      group <- filesplit[1, 2]
      samplename <- filesplit[1, 3]
      topn <- filesplit[1, 4]
      thre <- filesplit[1, 5]

      if (samplename %in% rownames(SSN19add1)) {
        edgelist <- data.frame(SSN19add1[samplename, ], row.names = samplename, check.names = FALSE)
        colnames(edgelist) <- colnames(SSN19add1)
        SSN_list[[file19]] <- edgelist
        edgelist <- data.frame(t(edgelist), check.names = FALSE) %>%
          rownames_to_column(var = 'edges_OTU') %>%
          separate(edges_OTU, into = c('V1', 'V2'), sep = ';.;')
        edgelist <- edgelist[!is.na(edgelist[samplename]), ]
        g <- graph_from_edgelist(as.matrix(edgelist[, c('V1', 'V2')]), directed = FALSE)
        E(g)$weight <- unlist(edgelist[samplename])
        E(g)$weight <- format(E(g)$weight, digits = 3)
        write_graph(g, file = paste0(mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'), format = "gml")
      } else {
        print(paste0('No file', mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'))
        G1 <- make_empty_graph(n = 0, directed = FALSE)
        write_graph(G1, file = paste0(mypath, phylotxt, '_', group, '_', samplename, '_', thre, '_', topn, '_05.gml'), format = "gml")
      }
    }

    SSNall <- data.table::rbindlist(SSN_list, fill = TRUE, use.names = TRUE, idcol = 'samplename') %>%
      melt(na.rm = TRUE)
    write.csv(SSNall, paste0(mypath, phylotxt, "_comselSSN.csv"), row.names = FALSE)

    meta <- meta(ps.ng.tax)
    meta2 <- meta
    row.names(meta2) <- gsub(row.names(meta2), pattern = 'X', replacement = '')
    meta <- rbind(meta, meta2)

    attrlist <- list()
    for (file in list.files(mypath, paste0(phylotxt, '.*.gml$'))) {
      G1 <- read_graph(paste0(mypath, file), format = "gml")
      tryCatch({
        E(G1)$weight <- abs(as.numeric(as.character(E(G1)$weight)))
        G_frame <- as.data.frame(list(
          mean(degree(G1)),
          average.path.length(G1),
          diameter(G1, unconnected = TRUE),
          mean(betweenness(G1)),
          mean(closeness(G1)),
          length(E(G1)),
          length(V(G1))
        ))
        colnames(G_frame) <- c('mean_degree', 'average_path', 'diameter', 'mean_betweenness', 'mean_closeness', 'edgenumber', 'nodenumber')
        rownames(G_frame) <- file
        attrlist[[file]] <- G_frame
      }, error = function(error_message) {
        message("Error is")
        print(file)
        attrlist[[file]] <- data.frame(mean_degree = 0, average_path = 0, diameter = 0, mean_betweenness = 0, mean_closeness = 0, edgenumber = 0, nodenumber = 0)
        return(NA)
      })
    }

    attrtable <- dplyr::bind_rows(attrlist, .id = 'gml')
    attrtable$samples <- str_split_fixed(rownames(attrtable), '_', 5)[, 3]
    attrtable <- attrtable %>%
      merge(meta, by.x = 'samples', by.y = 'row.names')
    write.csv(attrtable, paste0(mypath, phylotxt, "_type_attrtable_metaSSN.csv"), row.names = TRUE)
  }
}
