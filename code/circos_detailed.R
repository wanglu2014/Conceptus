####0817####
# Set working directory to data folder (relative to code folder)
setwd("../data")
dend <- DECIPHER::ReadDendrogram("test_tree_95")
# Use apply function for Min-Max normalization on each row
library(circlize)
library(ComplexHeatmap)
library(dendsort)
library(readxl)
dev.new()
####Deg###
data1 <- read_xlsx("Deg.xlsx")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Row names to add
new_rownames <- c("Erysipelotrichaceae incertae sedis", "Thomasclavelia", 
                  "Neglectibacter", "Merdimmobilis", "Fibrobacter")

# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
#### Plotting ###
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.105
track_height2 <- 0.075
track_height3 <- 0.03
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[, 1:7], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 1:7])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[, 10:14], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:14])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:9], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 8:9])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.3

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "Diabetes, Non-alcoholic fatty liver disease, T2DM and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.325, y = 0.9, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()


####OTU###

data1 <- read_xlsx("OTU.xlsx")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Assume data is your existing data frame

# Row names to add
new_rownames <- c("Enterobacter", 
                  "Erysipelotrichaceae incertae sedis", 
                  "Thomasclavelia", 
                  "Mitsuokella", 
                  "Anaeromassilibacillus", 
                  "Neglectibacter", 
                  "Merdimmobilis", 
                  "Lactococcus", 
                  "Adlercreutzia")

# Create new row data with NA values
new_rows <- matrix(NA, nrow = length(new_rownames), ncol = ncol(data))
colnames(new_rows) <- colnames(data)
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
#### Plotting ###
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.105
track_height2 <- 0.075
track_height3 <- 0.03
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[, 1:7], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 1:7])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[, 10:14], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:14])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:9], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 8:9])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)
# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.3

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })

# Draw legend
lgd1 <- Legend(title = "Diabetes, Non-alcoholic fatty liver disease, T2DM and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.325, y = 0.9, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()


####Deg+OTU###

data1 <- read_xlsx("Deg_OTU.xlsx")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Create a new data frame containing new rows to add
new_rows <- data.frame(matrix(NA, nrow = 4, ncol = ncol(data)))
rownames(new_rows) <- c("Erysipelotrichaceae incertae sedis", 
                        "Thomasclavelia", 
                        "Neglectibacter", 
                        "Merdimmobilis")
colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]

#### Plotting ###
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.105
track_height2 <- 0.075
track_height3 <- 0.03
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[, 1:7], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 1:7])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[, 10:14], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:14])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:9], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 8:9])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)
# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.3

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })

# Draw legend
lgd1 <- Legend(title = "Diabetes, Non-alcoholic fatty liver disease, T2DM and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.325, y = 0.9, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()






# Already in data folder from previous setwd
dend <- DECIPHER::ReadDendrogram("test_tree_95")
# Use apply function for Min-Max normalization on each row
library(circlize)
library(ComplexHeatmap)
library(dendsort)
library(readxl)

####0818treat_importance#####
####Deg###
data1 <- read_xlsx("Deg.xlsx")
data1[[2]] <- paste(data1[[2]], data1[[3]], sep = "_")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Row names to add
new_rownames <- c("Erysipelotrichaceae incertae sedis", "Thomasclavelia", 
                  "Neglectibacter", "Merdimmobilis", "Fibrobacter")

# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.144
track_height2 <- 0.112
track_height3 <- 0.032
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,11:19], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 19:11])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[,1:7], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 7:1])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:10], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:8])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)






# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "T2D and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()


####OTU###

data1 <- read_xlsx("OTU.xlsx")
data1[[2]] <- paste(data1[[2]], data1[[3]], sep = "_")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Assume data is your existing data frame

# Row names to add
new_rownames <- c("Enterobacter", 
                  "Erysipelotrichaceae incertae sedis", 
                  "Thomasclavelia", 
                  "Mitsuokella", 
                  "Anaeromassilibacillus", 
                  "Neglectibacter", 
                  "Merdimmobilis", 
                  "Lactococcus", 
                  "Adlercreutzia")


# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.144
track_height2 <- 0.112
track_height3 <- 0.032
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,11:19], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 19:11])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[,1:7], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 7:1])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:10], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:8])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)



# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "T2D and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()




####Deg+OTU###

data1 <- read_xlsx("Deg+OTU.xlsx")
data1[[2]] <- paste(data1[[2]], data1[[3]], sep = "_")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Create a new data frame containing new rows to add
new_rows <- data.frame(matrix(NA, nrow = 4, ncol = ncol(data)))
rownames(new_rows) <- c("Erysipelotrichaceae incertae sedis", 
                        "Thomasclavelia", 
                        "Neglectibacter", 
                        "Merdimmobilis")
# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row
data <- t(apply(data, 1, function(x) (x - min(x)) / (max(x) - min(x))))
data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("white", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.144
track_height2 <- 0.112
track_height3 <- 0.032
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,11:19], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 19:11])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[,1:7], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 7:1])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 8:10], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 10:8])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)






# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "T2D and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()


####0818treat_diff####
####Dif_type###
data1 <- read_xlsx("Dif_type.xlsx")
data1[[2]] <- paste(data1[[2]], data1[[3]], sep = "_")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Create a new row name vector
new_rownames <- c( "Gammaretrovirus",
                   "Enterobacter",                   
                   "Escherichia", 
                   "Neisseria",                   
                   "Bordetella",                    
                   "Parasutterella",                  
                   "Holdemania",                    
                   "Erysipelotrichaceae incertae sedis",
                   "Thomasclavelia",                    
                   "Megasphaera",                  
                   "Megamonas",                    
                   "Agathobacter",                  
                   "Tyzzerella",                    
                   "Fusicatenibacter",                 
                   "Anaeromassilibacillus",             
                   "Butyricicoccus",                  
                   "Neglectibacter",                    
                   "Merdimmobilis",                  
                   "Gemella",                    
                   "Gordonibacter")

# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames
colnames(new_rows) <- colnames(data)

# Add new rows to original data frame
data <- rbind(data, new_rows)# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row

data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#7E3334"))
col_fun2 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#546D93"))
col_fun3 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#647645"))
# Adjust track heights
track_height1 <- 0.128
track_height2 <- 0.096
track_height3 <- 0.032
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,1:8], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 8:1])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[,12:17], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 17:12])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 9:11], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 11:9])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)






# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "Diabetes", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Fat", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()

####Dif###

data1 <- read_xlsx("Dif.xlsx")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[1,]
data <- data[-1,]

# Create a new row name vector
new_rownames <- c("Gammaretrovirus", "Enterobacter", "Escherichia", "Neisseria",
                  "Bordetella", "Parasutterella", "Holdemania", 
                  "Erysipelotrichaceae incertae sedis", "Thomasclavelia", 
                  "Megasphaera", "Megamonas", "Agathobacter", "Tyzzerella", 
                  "Fusicatenibacter", "Anaeromassilibacillus", "Butyricicoccus", 
                  "Neglectibacter", "Merdimmobilis", "Gemella", "Gordonibacter")

# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames
colnames(new_rows) <- colnames(data)

# Add new rows to original data frame
data <- rbind(data, new_rows)# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)

row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))

row.names(data) <- row

data <- as.data.frame(data)
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#7E3334"))
col_fun2 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#546D93"))
col_fun3 <- colorRamp2(c(-3, 0, 3), c("white", "lightgrey", "#647645"))
# Adjust track heights
track_height1 <- 0.064
track_height2 <- 0.096
track_height3 <- 0.032
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,c(1,2,4,7)], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, c(7,4,2,1)])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[,c(3,5,6,9,11,12)], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, c(12,11,9,6,5,3)])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, c(8,11)], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, c(11,8)])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)






# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "Diabetes", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Fat", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()

####0901####
###dif_type
data1 <- read_xlsx("dif_type.xlsx")
data1[[2]] <- paste(data1[[2]], data1[[3]], sep = "_")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:3),]

# Create a new data frame containing new rows to add
new_rownames <- c(
  "Gammaretrovirus", "Enterobacter", "Escherichia", 
  "Bordetella", "Parasutterella", "Holdemania", 
  "Erysipelotrichaceae incertae sedis", "Thomasclavelia", "Megasphaera", 
  "Megamonas", "Agathobacter", "Tyzzerella", 
  "Fusicatenibacter", "Anaeromassilibacillus", "Butyricicoccus", 
  "Neglectibacter", "Merdimmobilis", "Gemella", 
  "Gordonibacter"
)
# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames
colnames(new_rows) <- colnames(data)

# Add new rows to original data frame
data <- rbind(data, new_rows)# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))

# Set row names of new data frame to new_rownames
rownames(new_rows) <- new_rownames

colnames(new_rows) <- colnames(data)
# Add new rows to original data frame
data <- rbind(data, new_rows)
row <- rownames(data)
data[data == ""] <- NA
data[is.na(data)] <- 0
data <- data.frame(lapply(data, as.numeric))
# Define MAD normalization function, keeping 0 unchanged
mad_scaler <- function(x) {
  non_zero_indices <- which(x != 0)  # Find positions of non-zero elements
  med <- median(x[non_zero_indices], na.rm = TRUE)
  mad_value <- mad(x[non_zero_indices], na.rm = TRUE)
  
  # If MAD is 0, avoid division by 0, return 0 directly
  if (mad_value == 0) {
    x[non_zero_indices] <- 0
  } else {
    x[non_zero_indices] <- (x[non_zero_indices] - med) / mad_value
  }
  
  return(x)
}

# Apply MAD Scaler to entire data frame
data <- apply(data, 2, mad_scaler) # Apply MAD normalization by column

# Convert processed data back to data frame
data <- as.data.frame(data)
rownames(data) <- row
data[data == "NaN"] <- 0

leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#647645"))
# Adjust track heights
track_height2 <- 0.091
track_height1 <- 0.117
track_height3 <- 0.039
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))


# Reorder data and labels according to dend
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[,1:9], col = col_fun1, cluster = FALSE, rownames.side = "outside",  
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 9:1])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)





# Draw second heatmap
circos.heatmap(data_ordered[,13:19], col = col_fun2,
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 19:13])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)



# Draw third heatmap
circos.heatmap(data_ordered[, 10:12], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 12:10])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)



# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.4

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })


# Draw legend
lgd1 <- Legend(title = "T2D and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.114, y = 0.89, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()


####difs###
# Read data and perform initial processing
data1 <- read_xlsx("difs.xlsx")
data <- t(data1)
data <- as.data.frame(data)
colnames(data) <- data[2,]
data <- data[-(1:2),]

# Define new row names
new_rownames <- c(
  "Gammaretrovirus", "Enterobacter", "Escherichia", 
  "Bordetella", "Parasutterella", "Holdemania", 
  "Erysipelotrichaceae incertae sedis", "Thomasclavelia", "Megasphaera", 
  "Megamonas", "Agathobacter", "Tyzzerella", 
  "Fusicatenibacter", "Anaeromassilibacillus", "Butyricicoccus", 
  "Neglectibacter", "Merdimmobilis", "Gemella", 
  "Gordonibacter"
)

# Create new data frame with NA values, rows matching new_rownames length, columns matching data
new_rows <- as.data.frame(matrix(NA, nrow = length(new_rownames), ncol = ncol(data)))
rownames(new_rows) <- new_rownames
colnames(new_rows) <- colnames(data)

# Add new rows to original data frame
data <- rbind(data, new_rows)

# Handle missing values, convert empty and NA to 0
data[data == ""] <- NA
data[is.na(data)] <- 0
# Restore row names
row <- rownames(data)
# Convert all data frame elements to numeric
data <- data.frame(lapply(data, as.numeric))

# Define MAD normalization function, keeping 0 unchanged
mad_scaler <- function(x) {
  non_zero_indices <- which(x != 0)  # Find positions of non-zero elements
  med <- median(x[non_zero_indices], na.rm = TRUE)
  mad_value <- mad(x[non_zero_indices], na.rm = TRUE)
  
  # If MAD is 0, avoid division by 0, return 0 directly
  if (mad_value == 0) {
    x[non_zero_indices] <- 0
  } else {
    x[non_zero_indices] <- (x[non_zero_indices] - med) / mad_value
  }
  
  return(x)
}

# Apply MAD Scaler to entire data frame
data <- apply(data, 2, mad_scaler) # Apply MAD normalization by column

# Convert processed data back to data frame
data <- as.data.frame(data)
rownames(data) <- row
# Select leaf labels and reorder data frame
# Assume dend is an existing dendrogram object
leaves_labels <- labels(dend)
data <- data[leaves_labels,]
## Plotting ##
# Define color mapping functions
col_fun1 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#7E3334"))
col_fun2 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#546D93"))
col_fun3 <- colorRamp2(c(-1, 0, 1), c("grey", "#F0F0F0", "#647645"))
# Adjust track heights
track_height1 <- 0.105
track_height2 <- 0.075
track_height3 <- 0.03
dend.track.height <- 0.075
label_track_height <- 0.1

# Clear previous plot area
grid.newpage()
circos.clear()

# Set initial parameters
circos.par(start.degree = 50, gap.after = c(40))
data_ordered <- data
labels <- rownames(data_ordered)  # Ensure label order matches sorted data

# Draw first heatmap
circos.heatmap(data_ordered[, 8:14], col = col_fun1, rownames.side = "outside",
               cluster = FALSE, rownames.cex = 0.4, 
               track.height = track_height1, cell.border = "white")

circos.track(track.index = get.current.track.index(), 
             panel.fun = function(x, y) {
               if (CELL_META$sector.numeric.index == 1) {
                 cn = colnames(data_ordered[, 14:8])
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                             1:n - 0.5, cn, 
                             cex = 0.23, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

# Draw second heatmap
circos.heatmap(data_ordered[, 1:5], col = col_fun2, cluster = FALSE,  
               track.height = track_height2, cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 5:1])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw third heatmap
circos.heatmap(data_ordered[, 6:7], col = col_fun3, 
               cluster = FALSE, 
               dend.track.height = dend.track.height,
               dend.side = "inside",
               track.height = track_height3, 
               cell.border = "white")

circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  if (CELL_META$sector.numeric.index == 1) {
    cn = colnames(data_ordered[, 7:6])
    n = length(cn)
    circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(0.5, "mm"), 
                1:n - 0.5, cn, 
                cex = 0.23, adj = c(0, 0.5), facing = "inside")
  }
}, bg.border = NA)

# Draw label track (check if dend is aligned)
#circos.trackPlotRegion(ylim = c(0, 1), bg.border = NA, track.height = label_track_height,
#                       panel.fun = function(x, y) {
#                         for (i in seq_len(length(labels))) {
#                           circos.text(i - 0.5, 0, labels[i], adj = c(0, 0.5),
#                                       cex = 0.4,  
#                                       facing = "clockwise", 
#                                       niceFacing = TRUE) 
#                         }
#                       })
# Draw circular dendrogram without text
max_height <- attr(dend, "height")*0.3

circos.trackPlotRegion(ylim = c(0, max_height), bg.border = NA,
                       track.height = dend.track.height, panel.fun = function(x, y) {
                         circos.dendrogram(dend, max_height = max_height)
                       })




# Draw legend
lgd1 <- Legend(title = "Diabetes, Non-alcoholic fatty liver disease, T2DM and NAFLD", col_fun = col_fun1)
pushViewport(viewport(x = 0.325, y = 0.9, width = 0.18, height = 0.18)) 
grid.draw(lgd1)
popViewport()

lgd2 <- Legend(title = "IBD", col_fun = col_fun2)
pushViewport(viewport(x = 0.06, y = 0.5, width = 0.18, height = 0.18)) 
grid.draw(lgd2)
popViewport()

lgd3 <- Legend(title = "Obese", col_fun = col_fun3)
pushViewport(viewport(x = 0.06, y = 0.2, width = 0.18, height = 0.18)) 
grid.draw(lgd3)
popViewport()




# Find elements in leaves_label but not in data frame row names
diff_in_leaves <- setdiff(leaves_labels, rownames(data))

# Find elements in data frame row names but not in leaves_label
diff_in_rownames <- setdiff(rownames(data), leaves_labels)

# Find common elements of both sets
common_elements <- intersect(leaves_labels, rownames(data))

# Print results
cat("Elements in leaves_label but not in data frame row names:\n")
print(diff_in_leaves)

cat("Elements in data frame row names but not in leaves_label:\n")
print(diff_in_rownames)

cat("Common elements in both:\n")
print(common_elements)