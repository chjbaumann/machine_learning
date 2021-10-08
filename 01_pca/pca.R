### SET PATH ###

setwd("/Users/Christian/Desktop/ML/01_pca/")

### LOAD LIBRARIES ###

library("maptools")

### IMPORT DATA ###

df <- read.csv("./input/distances.csv", row.names = 1)

### DISTANCES RELATIVE TO EARTH'S HALF CIRCUMFERENCE (MAXIMUM POSSIBLE DISTANCE) ###

df <- df/(20000)

##############################################################################################################
### PCA ######################################################################################################
##############################################################################################################

pca <- prcomp(df, center = FALSE, scale. = FALSE)

### LABELS FOR PCA AXES ###

ax_labels <- c()
for (i in c(1:81)){
  ax_labels <- c(ax_labels, paste("PC", i, " (", round(summary(pca)[[6]][2,i]*100, 2), "%)", sep = ""))
}

### PCA SCORE PLOTS ###

cairo_pdf("./output/pca.pdf", 8, 8)

par(mfrow = c(2,2), family = "serif")

plot(pca$x[,1], pca$x[,2], las = 1, pch = 21, bg = "grey",
     main = "PCA 1", xlab = ax_labels[1], ylab = ax_labels[2])
pointLabel(pca$x[,1], pca$x[,2], rownames(df), cex = 0.6)
box(lwd = 1.2)

plot(pca$x[,2], pca$x[,3], las = 1, pch = 21, bg = "grey",
     main = "PCA 2", xlab = ax_labels[2], ylab = ax_labels[3])
pointLabel(pca$x[,2], pca$x[,3], rownames(df), cex = 0.6)
box(lwd = 1.2)

plot(pca$x[,3], pca$x[,4], las = 1, pch = 21, bg = "grey",
     main = "PCA 3", xlab = ax_labels[3], ylab = ax_labels[4])
pointLabel(pca$x[,3], pca$x[,4], rownames(df), cex = 0.6)
box(lwd = 1.2)

plot(pca$x[,4], pca$x[,5], las = 1, pch = 21, bg = "grey",
     main = "PCA 4", xlab = ax_labels[4], ylab = ax_labels[5])
pointLabel(pca$x[,4], pca$x[,5], rownames(df), cex = 0.6)
box(lwd = 1.2)

dev.off()

##############################################################################################################
### PROPORTION OF VARIANCE ###################################################################################
##############################################################################################################

cairo_pdf("./output/pca_variance.pdf", 9, 4)

par(mfrow = c(1,2), family = "serif")

plot(c(1:81), summary(pca)[[6]][2,]*100, las = 1, type = "b", pch = 19, xlim = c(1,10),
     main = "Proportion of Variance", xlab = "Principal Component", ylab = "Proportion [%]")
box(lwd = 1.2)

plot(c(1:81), summary(pca)[[6]][3,]*100, las = 1, type = "b", pch = 19, xlim = c(1,10), ylim = c(75,110),
     main = "Cumulative Proportion", xlab = "Principal Component", ylab = "Cumulative Proportion [%]")
abline(h = 99.5, col = "red")
text(1.5, 102, "99.5 %", col = "red")
box(lwd = 1.2)

dev.off()

##############################################################################################################
### EXPORT PC 1-4 ############################################################################################
##############################################################################################################

write.csv(pca$x[,1:4], "./output/distances_pca.csv")

##############################################################################################################
### PC1 VS SUM OF DISTANCES ##################################################################################
##############################################################################################################

cairo_pdf("./output/pca_pc1.pdf", 4, 4)

par(mfrow = c(1,1), family = "serif")

plot(rowSums(df), pca$x[,1], pch = 21, bg = "grey", las = 1,
     main = "PC1 vs Sum of Distances", xlab = "Sum of Distances", ylab = "PC1")
box(lwd = 1.2)

dev.off()

