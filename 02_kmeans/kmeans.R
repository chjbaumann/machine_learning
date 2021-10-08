### SET PATH ###

setwd("/Users/Christian/Desktop/ML/02_kmeans/")

### LOAD LIBRARIES ###

library("factoextra")
library("maptools")

### IMPORT DATA ###

df <- read.csv("./input/distances_pca.csv", row.names = 1)

##############################################################################################################
### K-MEANS ##################################################################################################
##############################################################################################################

### FIND OPTIMUM NUMBER OF CLUSTERS ###

fviz_nbclust(df, kmeans, method = "silhouette")

### K-MEANS ###

kdat <- kmeans(df, 3, iter.max = 1000, nstart = 100)

### ADD CLUSTERS TO DATAFRAME ###

df <- cbind(df, kdat$cluster)

colnames(df)[5] <- "Cluster"

### CENTERS OF CLUSTERS ###

centers <- kdat$centers

### EXPORT DATAFRAME ###

write.csv(df, "./output/distances_pca_clusters.csv")

##############################################################################################################
### PCA SCORE PLOTS WITH CLUSTERS ############################################################################
##############################################################################################################

cluster_colors <- c("#91ee92", "skyblue", "#efa9f5")

cairo_pdf("./output/pca_clusters.pdf", 8, 8)

par(mfrow = c(2,2), family = "serif")

plot(df[,1], df[,2], las = 1, pch = 21, bg = cluster_colors[df[,5]],
     main = "PCA 1", xlab = "PC1", ylab = "PC2")
points(centers[,1], centers[,2], pch = 23, cex = 1.5, bg = c("darkgreen", "blue", "red"))
pointLabel(df[,1], df[,2], rownames(df), cex = 0.6)
box(lwd = 1.2)

plot(df[,2], df[,3], las = 1, pch = 21, bg = cluster_colors[df[,5]],
     main = "PCA 2", xlab = "PC2", ylab = "PC3")
points(centers[,2], centers[,3], pch = 23, cex = 1.5, bg = c("darkgreen", "blue", "red"))
pointLabel(df[,2], df[,3], rownames(df), cex = 0.6)
box(lwd = 1.2)

plot(df[,3], df[,4], las = 1, pch = 21, bg = cluster_colors[df[,5]],
     main = "PCA 3", xlab = "PC3", ylab = "PC4")
points(centers[,3], centers[,4], pch = 23, cex = 1.5, bg = c("darkgreen", "blue", "red"))
pointLabel(df[,3], df[,4], rownames(df), cex = 0.6)
box(lwd = 1.2)

dev.off()

