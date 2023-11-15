# Group 9 Final Project: Bankruptcy Prediction
# ----------------------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------------------
library(readr)
library(tidyverse)
library(creditmodel) 
library(caret)
library(smotefamily)
library(dplyr)
library(factoextra)
library(caret)
library(tidyverse)
library(e1071)
library(glmnet)
library(class)

# ----------------------------------------------------------------------------------------
# Part I. Data Preparation
# ----------------------------------------------------------------------------------------
## Read the data set in R.
data <- read_csv("data.csv")

## For dependent variable - Bankruptcy: rename it and convert it from numeric to factor.
colnames(data)[1] <- "Bankruptcy"

data$Bankruptcy <- factor(data$Bankruptcy, levels = c(1, 0))
levels(data$Bankruptcy) <- c("Yes", "No")

## Set the seed to 97429
set.seed(97429)
## Partition the data set (train-70%, test-30%)
inTrain <- sample(nrow(data), 0.7*nrow(data))
train <- data.frame(data[inTrain, ])
test <- data.frame(data[-inTrain, ])

# Explore1: Visualize the dependent variable in train data set
plot(data$Bankruptcy, xlab = "Bankrupt?", ylab = "Number of Observations", main = "Unbalanced Dataset")
## Conclusion: Unbalanced data set

# Explore2: Logistic Regression Model
## Run a logistic regression model of Bankruptcy on all independent variables.
fit0 <- glm(Bankruptcy ~ ., data = train, family = "binomial")
summary(fit0)
## Use a cutoff of 0.5 and do the classification. 
cutoff <- 0.5
## For in-sample predictions / Use train data set
predicted.probability.train <- predict(fit0, data = train, type = "response") ### type = "response" gives the predicted probabilities.
predicted.class.train <- ifelse(predicted.probability.train > cutoff, "No", "Yes")
predicted.class.train <- factor(data.frame(predicted.class.train)$predicted.class.train, levels = c("Yes", "No"))
## For out-sample predictions / Use test data set
predicted.probability.test <- predict(fit0, newdata = test, type = "response")
predicted.class.test <- ifelse(predicted.probability.test > cutoff, "No", "Yes")
predicted.class.test <- factor(data.frame(predicted.class.test)$predicted.class.test, levels = c("Yes", "No"))
## Compute the confusion matrix for both in-sample and out-sample predictions.
## For in-sample predictions / Use train data set
actual_train <- train$Bankruptcy
(CM_train <- table(actual_train, predicted.class.train)) ### confusion matrix
(Acc_train <- (CM_train[1, 1]+CM_train[2, 2])/sum(CM_train)) ### accuracy
(Sen_train <- CM_train[1, 1]/(CM_train[1, 1]+CM_train[1, 2])) ### sensitivity
## For out-sample predictions / Use test data set
actual_test <- test$Bankruptcy
(CM_test <- table(actual_test, predicted.class.test)) ### confusion matrix
(Acc_test <- (CM_test[1, 1]+CM_test[2, 2])/sum(CM_test)) ### accuracy
(Sen_test <- CM_test[1, 1]/(CM_test[1, 1]+CM_test[1, 2])) ### sensitivity
## Conclusion: high accuracy but low sensitivity.

# Dimensionality Reduction
## Method 1: Low Variance Filter
data_lvf <- low_variance_filter(data[-1], lvp = 0.97)
data <- cbind(data[1], data_lvf)
## Conclusion: remove 2 categorical variables

## Method 2: High Correlation Filter
### Correlation Matrix
correlationMatrix <- cor(data[-1])
### filter out variables whose correlation coefficient is higher than 0.8.
(highcorrelation <- findCorrelation(correlationMatrix, cutoff = 0.8)) # return an index vector
data <- data[ , -(highcorrelation+1)]
## Conclusion: Remove 23 variables

## Method 3: Principal Component Analysis (Optional)
library(factoextra)
PCA <- function(train_scale, test_scale = FALSE) {
  # Compute PCA
  pca <- prcomp(train_scale[-1], scale = FALSE)
  # Visualize Cumulative Proportions of Explained Variances 
  plot(summary(pca)$importance[3,], ylab = "Cumulative Proportion of Explained Variance")
  # Select k
  k = min(which(summary(pca)$importance[3,]>0.95))
  # Predict with PCA for train data set
  train_pca <- data.frame(predict(pca, newdata = train_scale))
  # Build a new train data set
  train_pca <- cbind(train_scale[1], train_pca[1:k])
  # If test_scale = FALSE, only return the train_pca
  if (!("data.frame" %in% class(test_scale))) {
    return(train_pca)
  }
  # If test_scale is one of inputs, return a list including train_pca and test_pca.
  if ("data.frame" %in% class(test_scale)) {
    test_pca <- data.frame(predict(pca, newdata = test_scale))
    test_pca <- cbind(test_scale[1], test_pca[1:k])
    return(list(train_pca, test_pca))
  }
}

# -------------------------------------------------------------------------------------------
# Part II. Algorithm Selection
# -------------------------------------------------------------------------------------------
## K-Folds Cross-validation
### For evaluate generalized error, we divide the data set into 4 different parts and test our model on one part for four times, then we will compute its average error.
folds <- createFolds(data$Bankruptcy, k = 4)

## Create a normalization function
Normalize <- function(train, test=FALSE) {
  train_scale <- scale(train[ , -1])
  mean <- t(data.frame(attr(train_scale, "scaled:center")))
  std <- t(data.frame(attr(train_scale, "scaled:scale")))
  train_scale <- cbind(train[1], train_scale)
  if (!("data.frame" %in% class(test))) {
    return (train_scale)
  }
  if ("data.frame" %in% class(test)) {
    mean <- mean[rep(1, nrow(test)), ]
    std <- std[rep(1, nrow(test)), ]
    test_scale <- (test[ , -1]-mean)/std
    test_scale <- cbind(test[1], test_scale)
    return (list(train_scale, test_scale))
  }
}

## Create a smote function - solve the unbalanced data set problem
Smote <- function(train_scale, S = 0) {
  # Visualize dependent variable before smote
  # plot(train_scale$Bankruptcy)
  # SMOTE
  smote_train <- BLSMOTE(train_scale[-1], train_scale$Bankruptcy, dupSize = S, K = 5, C = 5, method = "type1")$data
  # move the last column (dependent variable) to the first position
  train_scale <- select(smote_train, "class", everything())
  # rename it
  colnames(train_scale)[1] <- "Bankruptcy"
  # convert it from character to factor
  train_scale$Bankruptcy <- factor(train_scale$Bankruptcy, levels = c("Yes", "No"))
  # Visualize dependent variable after smote
  # plot(train_scale$Bankruptcy)
  return(train_scale)
}



## 1. Logistic Regression with Regularization
set.seed(97429)
# accuracy vector for glm
glm.acc <- rep(0, length(folds))
# sensitivity vector for glm
glm.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  # cross validation for each loop
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  # normalize the data using defined function
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  # oversample training data set using defined function
  train_scale <- Smote(train_scale)
  # train our logistic regression model, optimize it with ROC metric
  # tune grid for lambda 0~0.5, alpha 0~1
  model.glm <- train(Bankruptcy ~ .,
                     data = train_scale,
                     method = 'glmnet',
                     metric = "ROC",
                     maximize = TRUE,
                     # tune glm with 2-times 5-folds cross validation
                     trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                summaryFunction=twoClassSummary, classProbs=TRUE),
                     tuneGrid = expand.grid(lambda = seq(0, 0.5, by=0.01),
                                            alpha = seq(0, 1, by=0.1)))
  
  pred.test <- predict(model.glm, test_scale)
  # Compute the confusion matrix
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  # accuracy of glm for i fold
  glm.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  # sensitivity of glm for i fold
  glm.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}


## 2. K-Nearest Neighbor
set.seed(97429)
# accuracy vector for KNN
knn.acc <- rep(0, length(folds))
# sensitivity vector for KNN
knn.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  # cross validation for each loop
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  # normalize the data using defined function
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  # oversample training data set using defined function
  train_pca <- Smote(train_pca)
  # train our K-Nearest Neighbor model, optimize it with ROC metric
  # tune grid for k 1~20
  model.knn <- train(Bankruptcy ~ .,
                     data = train_pca,
                     method = "knn",
                     metric = "ROC",
                     maximize = TRUE,
                     # tune KNN with 2-times 5-folds cross validation
                     trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                 summaryFunction = twoClassSummary, classProbs=TRUE),
                     tuneGrid = expand.grid(k = 1: 20))
  
  pred.test <- predict(model.knn, test_pca)
  # Compute the confusion matrix
  CM.test <- table(test_pca$Bankruptcy, pred.test)
  # accuracy of KNN for i fold
  knn.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  # sensitivity of KNN for i fold
  knn.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}

## 3. Random Forest
# Random forest
set.seed(97429)
# accuracy vector for random forest
parRF.acc <- rep(0, length(folds))
# sensitivity vector for random forest
parRF.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  # cross validation for each loop
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  # normalize the data using defined function
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  # oversample training data set using defined function
  train_scale <- Smote(train_scale)
  # train our Random Forest model, optimize it with ROC metric
  model.parRF <- train(x=train_scale[-1], y=train_scale$Bankruptcy,
                       method = "parRF",
                       metric = "ROC",
                       maximize = TRUE,
                       # tune random forest with 2-times 5-folds cross validation
                       trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                        summaryFunction=twoClassSummary, classProbs=TRUE))
  
  pred.test <- predict(model.parRF, test_scale)
  # Compute the confusion matrix
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  # accuracy of random forest for i fold
  parRF.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  # sensitivity of random forest for i fold
  parRF.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}

# Boosted Generalized Linear Model
set.seed(97429)
glmboost.acc <- rep(0, length(folds))
glmboost.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  train_scale <- Smote(train_scale)
  model.glmboost <- train(Bankruptcy ~ .,
                          data=train_scale,
                          method = "glmboost",
                          metric = "ROC",
                          maximize = TRUE,
                          trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                  summaryFunction=twoClassSummary, classProbs=TRUE))
  
  pred.test <- predict(model.glmboost, test_scale)
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  glmboost.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  glmboost.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}

# Cost-Sensitive CART
set.seed(97429)
rpart.acc <- rep(0, length(folds))
rpart.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  train_scale <- Smote(train_scale)
  model.rpart <- train(x=train_scale[-1], y=train_scale$Bankruptcy,
                       method = "rpartCost",
                       trControl = trainControl(method="repeatedcv", number=5, repeats=2))
  
  pred.test <- predict(model.rpart, test_scale)
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  rpart.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  rpart.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}

# Penalized Discriminant Analysis
set.seed(97429)
pda.acc <- rep(0, length(folds))
pda.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  train_scale <- Smote(train_scale)
  model.pda <- train(Bankruptcy ~ .,
                     data=train_scale,
                     method = "pda",
                     metric = "ROC",
                     maximize = TRUE,
                     trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                              summaryFunction=twoClassSummary, classProbs=TRUE),
                     tuneGrid = expand.grid(lambda=seq(0,1,by=0.1)))
  pred.test <- predict(model.pda, test_scale)
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  pda.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  pda.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}

# Deep Neural Network
set.seed(97429)
dnn.acc <- rep(0, length(folds))
dnn.sen <- rep(0, length(folds))
for (i in 1:length(folds)) {
  train.data = data[-folds[[i]], ]
  test.data = data[folds[[i]], ]
  data_scale <- Normalize(train.data, test.data)
  train_scale <- data_scale[[1]]
  test_scale <- data_scale[[2]]
  train_scale <- Smote(train_scale)
  # train our Deep Neural Network model, optimize it with ROC metric
  model.dnn <- train(Bankruptcy ~ .,
                     data=train_scale,
                     method = "dnn",
                     metric = "ROC",
                     maximize = TRUE,
                     trControl = trainControl(method="repeatedcv", number=5, repeats=2,
                                              summaryFunction=twoClassSummary, classProbs=TRUE))
  
  pred.test <- predict(model.dnn, test_scale)
  CM.test <- table(test_scale$Bankruptcy, pred.test)
  dnn.acc[i] <- (CM.test[1,1]+CM.test[2,2])/sum(CM.test)
  dnn.sen[i] <- CM.test[1, 1]/(CM.test[1, 1]+CM.test[1, 2])
}
