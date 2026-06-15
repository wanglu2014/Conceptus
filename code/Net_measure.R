```{r}
library(dplyr)
library(readr)
library(stringr)
library(data.table)
Class_conticlass_metrics = function(meta_SSN, Prednum_list, True2list) {
  measure_list = list()

  for (typename in unique(meta_SSN$type)) {
    phylo_metaSSNtype = meta_SSN %>% filter(type == typename)

    for (Truename in True2list) {
      for (Pred_num in Prednum_list) {
        # Remove rows with NA values
        phylo_metaSSN = phylo_metaSSNtype %>% filter(!is.na(.[[Truename]]))

        # Check if more than binary classification
        unique_levels = unique(phylo_metaSSN[[Truename]])
        if (length(unique_levels) > 2) {
          next
        }

        # Convert to factor and set levels
        phylo_metaSSN[[Truename]] = as.factor(phylo_metaSSN[[Truename]])
        levels(phylo_metaSSN[[Truename]]) = c(1, 2)
        phylo_metaSSN[[Truename]] = as.numeric(phylo_metaSSN[[Truename]])

        aucvalue = 'NA'
        phylo_metaSSN[[Pred_num]] = as.numeric(phylo_metaSSN[[Pred_num]])

        tryCatch(
          {
            rocdemo = pROC::roc(predictor = phylo_metaSSN[[Pred_num]], response = phylo_metaSSN[[Truename]])
            aucvalue = as.numeric(rocdemo$auc)
          },
          error = function(error_message) {
            message(paste0("Error in ", Pred_num, " ", Truename))
            return(aucvalue)
          }
        )

        measure_list[[paste0(typename, '-', Pred_num, '-', Truename)]] = rbind(
          data.frame(value = aucvalue, row.names = 'AUC'),
          data.frame(value = median(unique(phylo_metaSSN$edgenumber)), row.names = 'edgenumber'),
          data.frame(value = length(unique(phylo_metaSSN$samples)), row.names = 'samplesize')
        ) %>% rownames_to_column(var = 'names')
      }
    }
  }

  measuretable = rbindlist(measure_list, use.names = TRUE, fill = TRUE, idcol = 'groupname')
  return(measuretable)
}
# Specify directory path (relative to code folder)
directory_path <- "../data/"
# List all files with type_attrtable_metaSSN_0520.csv suffix
# Note: These intermediate files are not included in the package.
# The final aggregated data is in Supplementary_Table_4_Network_Attributes.xlsx
file_list <- list.files(path = directory_path, pattern = "type_attrtable_metaSSN_0520.csv$", full.names = TRUE)
# Read all files and merge into one data frame
combined_df <- lapply(file_list, read_csv)
combined_table=rbindlist(combined_df, idcol = "filename",fill = T)
graphlist_table <- combined_table %>%
  mutate(
    group = str_extract(gml, "(?<=_)[^_]+_[^_]+(?=_\\d{4}\\.gml$)"),
    phylotxt = str_split_fixed(gml, "_", 3)[, 1],
    samples = str_split_fixed(gml, "_", 4)[, 3]
  )
write.csv(graphlist_table, paste0(directory_path, "graphlist_table.csv"))
phylomeasure_list=list()
for (group_name in unique(graphlist_table$group)) {
  group_data <- graphlist_table %>% filter(group == group_name)

  for (phylotxtname in unique(group_data$phylotxt)) {
    phylo_mtypedegree <- as.data.frame(group_data %>% filter(phylotxt == phylotxtname),check.names = F)
    Trueclass_list=colnames(phylo_mtypedegree)[grepl(pattern = 'response*',x = colnames(phylo_mtypedegree))]
    Truenum_list=colnames(phylo_mtypedegree)[grepl(pattern = 'respnum*',x = colnames(phylo_mtypedegree))]

    #Predclass_list=colnames(meta_SSN)[grepl(pattern = 'cluster*',x = colnames(meta_SSN))]
    Prednum_list=c('shannon','edgenumber','nodenumber','mean_degree')#,'respnum.shannon.star','respnum.inverse_simpson.star')
    measured_conticlass=data.frame('NA')
    measured_conticonti=data.frame('NA')
    #measured_classclass='NA'

    measured_conticlass=Class_conticlass_metrics(phylo_mtypedegree,Prednum_list,Trueclass_list)
    #if(!is_empty(Truenum_list)){
    #measured_conticonti=Class_conticonti_metrics(phylo_mtypedegree,Prednum_list,Truenum_list)}
    #if(!is_empty(Trueclass_list)&!is_empty(Predclass_list)){
    #measured_classclass=Class_classclass_metrics(meta_SSN,Predclass_list,Trueclass_list)}
    phylomeasure_list[[paste0(phylotxtname,'__',group_name)]]=rbindlist(list(measured_conticlass,measured_conticonti),fill = T)
  }
}

phylomeasure_table=rbindlist(phylomeasure_list,use.names = T,fill = T,idcol = 'filename')
write.csv(phylomeasure_table, paste0(directory_path, 'Net_measure_0525.csv'))

```

```{r}
# Note: Net_measure_0529.csv is the output from the above analysis.
# For reproduction, run the above code block first or use pre-computed data.
phylomeasure_table=read.csv(paste0(directory_path, "Net_measure_0529.csv"))
# Custom function to split groupname column into grouptype, groupvar, groupresp
split_groupname <- function(groupname) {
  # Find positions of specific strings
  if (str_detect(groupname, "shannon")) {
    parts <- str_split(groupname, "-")[[1]]
    return(c(parts[1], "shannon", parts[3]))
  } else if (str_detect(groupname, "edgenumber")) {
    parts <- str_split(groupname, "-")[[1]]
    return(c(parts[1], "edgenumber", parts[3]))
  } else if (str_detect(groupname, "nodenumber")) {
    parts <- str_split(groupname, "-")[[1]]
    return(c(parts[1], "nodenumber", parts[3]))
  } else if (str_detect(groupname, "mean_degree")) {
    parts <- str_split(groupname, "-")[[1]]
    return(c(parts[1], "mean_degree", parts[3]))
  }
  return(c(NA, NA, NA))
}
library(tidyverse)
# Split groupname column and create new columns
df <- phylomeasure_table%>%
  mutate(
    groupvar = str_extract(groupname, "(shannon|edgenumber|nodenumber|mean_degree)"),
    grouptype = str_extract(groupname, paste0("^.+?(?=-", groupvar, ")")),
    groupresp = str_extract(groupname, "(?<=-)(response.*$)")
  )
openxlsx::write.xlsx(df, paste0(directory_path, "Net_measure_tab_0529.xlsx"))
```
