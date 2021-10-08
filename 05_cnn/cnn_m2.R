# According to https://tensorflow.rstudio.com/tutorials/advanced/images/cnn/
# & https://rpubs.com/HeatWave2019/537744

### SET PATH ###

setwd("/Users/chrjb/Desktop/05_cnn/")

### LOAD LIBRARIES ###

library("tensorflow")
library("keras")

##############################################################################################################
### IMPORT PICTURES ##########################################################################################
##############################################################################################################

cifar <- dataset_cifar10()


class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

index <- 1:16

par(mfcol = c(4,4), mar = rep(1, 4), family = "serif")
cifar$train$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})

##############################################################################################################
### BUILD MODEL ##############################################################################################
##############################################################################################################

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3), padding = "same") %>% 
  layer_conv_2d(filters  = 32, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(0.25) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                padding = "same") %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(0.25)

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dropout(0.5) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

##############################################################################################################
### TRAIN MODEL ##############################################################################################
##############################################################################################################

history <- model %>% 
  fit(
    x = cifar$train$x, y = cifar$train$y,
    epochs = 30,
    validation_data = unname(cifar$test),
    verbose = 2
  )


pdf("./results/training_m2.pdf", 8, 4)
par(mfrow = c(1,2), family = "serif")

plot(history$metrics$loss, type = "b", las = 1, pch = 21, bg = "dodgerblue", ylim = c(0.7, 2),
     main = "Loss", xlab = "Epoch", ylab = "Loss")
points(history$metrics$val_loss, type = "b", pch = 21, bg = "orange")
legend("topright", c("Training", "Validation"), pt.bg = c("dodgerblue", "orange"), pch = 21, bty = "n")
box(lwd = 1.2)

plot(history$metrics$accuracy, type = "b", las = 1,  ylim = c(0.2,0.8), pch = 21, bg = "dodgerblue",
     main = "Accuracy", xlab = "Epoch", ylab = "Accuracy")
points(history$metrics$val_accuracy, type = "b", pch = 21, bg = "orange")
box(lwd = 1.2)

dev.off()

##############################################################################################################
### PREDICTIONS ##############################################################################################
##############################################################################################################

predictions <- model %>% predict_classes(cifar$test$x[index,,,])

par(mfcol = c(4,4), mar = rep(1, 4), family = "serif")
cifar$test$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[predictions + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})




