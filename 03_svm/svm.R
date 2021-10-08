### SET PATH ###

setwd("/Users/Christian/Desktop/ML/03_svm")

### LOAD LIBRARIES ###

library("e1071")
library("maptools")

### IMPORT DATA ###

df <- read.csv("./input/training_data.csv", row.names = 1)

test_df <- read.csv("./input/test_data.csv", row.names = 1)

##############################################################################################################
### REARRANGE DATA ###########################################################################################
##############################################################################################################

# 12 means that cluster 1 & 2 are combined and 23 means that cluster 2 & 3 are combined

df12 <- df[,c(2,3,5)]
df23 <- df[,c(2,3,5)]

# clusters 12 and 23 are renamed as -1 and clusters 3 and 1 (respectively) as +1

for (i in c(1:dim(df)[1])){
  if (df12[i,3] == 3){df12[i,3] <- 1}else{df12[i,3] <- -1}
  if (df23[i,3] != 1){df23[i,3] <- -1}
}

##############################################################################################################
### SVM CALCULATION ##########################################################################################
##############################################################################################################

svmfit12 = svm(df12[,3] ~ ., data = df12, kernel = "linear", cost = 10, scale = FALSE)
svmfit23 = svm(df23[,3] ~ ., data = df23, kernel = "linear", cost = 10, scale = FALSE)

# EXTRACTION Of CLUSTER BORDERS

beta_12 = drop(t(svmfit12$coefs) %*% as.matrix(df12[svmfit12$index,c(1,2)]))
beta0_12 = svmfit12$rho
beta_23 = drop(t(svmfit23$coefs) %*% as.matrix(df23[svmfit23$index,c(1,2)]))
beta0_23 = svmfit23$rho

##############################################################################################################
### PLOTS ####################################################################################################
##############################################################################################################

cluster_colors <- c("#91ee92", "skyblue", "#efa9f5")

cairo_pdf("./output/svm_classification.pdf", 8, 4)

par(mfrow = c(1,2), family = "serif")

plot(df[,2], df[,3], las = 1, pch = 21, bg = cluster_colors[df[,5]],
     main = "SVM Classification", xlab = "PC2", ylab = "PC3")
abline(beta0_12 / beta_12[2], -beta_12[1] / beta_12[2])
abline(beta0_23 / beta_23[2], -beta_23[1] / beta_23[2])
box(lwd = 1.2)

plot(df[,2], df[,3], las = 1, pch = 21, bg = cluster_colors[df[,5]],
     main = "SVM Classification", xlab = "PC2", ylab = "PC3")
abline(beta0_12 / beta_12[2], -beta_12[1] / beta_12[2])
abline(beta0_23 / beta_23[2], -beta_23[1] / beta_23[2])
points(test_df[,2], test_df[,3], pch = 23, bg = "grey")
pointLabel(test_df[,2], test_df[,3], rownames(test_df), cex = 0.8)
box(lwd = 1.2)

dev.off()

