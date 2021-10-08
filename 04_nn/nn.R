### SET PATH ###

setwd("/Users/Christian/Desktop/ML/04_nn/")

### LOAD LIBRARIES ###

library("neuralnet")

### IMPORT DATA ###

df <- read.csv("./input/training_data.csv", row.names = 1)

test_df <- read.csv("./input/test_data.csv", row.names = 1)

##############################################################################################################
### DATA REARRANGEMENT AND Max - Min NORMALIZATION ###########################################################
##############################################################################################################

ndf <- matrix(NA, dim(df)[1], dim(df)[2]+2)
rownames(ndf) <- rownames(df)
colnames(ndf) <- c(colnames(df)[-5], "Cluster_1", "Cluster_2", "Cluster_3")

test_ndf <- matrix(NA, dim(test_df)[1], dim(test_df)[2])
rownames(test_ndf) <- rownames(test_df)
colnames(test_ndf) <- colnames(test_df)

for (i in c(1:4)){
  
  max_value <- max(df[,i], test_df[,i])
  min_value <- min(df[,i], test_df[,i])
  
  ndf[,i] <- (df[,i] - min_value) / (max_value - min_value)
  test_ndf[,i] <- (test_df[,i] - min_value) / (max_value - min_value)
  
}

##############################################################################################################
### DATA PREPARATION (CLUSTER REPRESENTATION) ################################################################
##############################################################################################################

for (i in c(1:dim(df)[1])){
  if (df[i,5] == 1){ndf[i,5:7] <- c(1,0,0)}
  if (df[i,5] == 2){ndf[i,5:7] <- c(0,1,0)}
  if (df[i,5] == 3){ndf[i,5:7] <- c(0,0,1)}
}

##############################################################################################################
### NEURAL NETWORK TRAINING ##################################################################################
##############################################################################################################

nn_simple <- neuralnet(Cluster_1+Cluster_2+Cluster_3 ~ PC1+PC2+PC3+PC4, data = ndf, hidden = 2,
                       act.fct = "logistic", err.fct = "sse", linear.output = FALSE, rep = 1)

nn_complex <- neuralnet(Cluster_1+Cluster_2+Cluster_3 ~ PC1+PC2+PC3+PC4, data = ndf, hidden = c(5,5),
                        act.fct = "logistic", err.fct = "sse", linear.output = FALSE, rep = 1)

##############################################################################################################
### NEURAL NETWORK APPLICATION ###############################################################################
##############################################################################################################

result_simple <- compute(nn_simple, test_ndf)$net.result
result_complex <- compute(nn_complex, test_ndf)$net.result

### NORMALIZATION ###
for (i in c(1:4)){
  result_simple[i,] <- 100*result_simple[i,]/sum(result_simple[i,])
  result_complex[i,] <- 100*result_complex[i,]/sum(result_complex[i,])
}

##############################################################################################################
### PLOTS ####################################################################################################
##############################################################################################################

cluster_colors <- c("#91ee92", "skyblue", "#efa9f5")


cairo_pdf("./output/NN_results.pdf", 8, 4)

par(mfrow = c(1,2), family = "serif")

barplot(t(result_simple), las = 2, yaxs = "i", ylim = c(0,100), col = cluster_colors,
        main = "Simple NN", ylab = "Probability per Cluster [%]")
box(lwd = 1.2)

barplot(t(result_complex), las = 2, yaxs = "i", ylim = c(0,100), col = cluster_colors,
        main = "Complex NN", ylab = "Probability per Cluster [%]")
box(lwd = 1.2)

dev.off()

##############################################################################################################
### PLOT NETWORKS ############################################################################################
##############################################################################################################

plot(nn_simple)

plot(nn_complex)

