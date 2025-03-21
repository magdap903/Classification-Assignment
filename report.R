## ----------------------------------------------------------------------------------------------------------------------------------
library(reshape2)
library(rsample)
library(ggplot2)
library(recipes)
library(caret)
library(lattice)
library(MASS)
library(rpart)
library(randomForest)
library(corrplot)
library(e1071)
library(pROC)
library(Metrics)




## ----------------------------------------------------------------------------------------------------------------------------------
heart <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv", show_col_types = FALSE)
head(heart)
colnames(heart)
#checking for any missing values
colSums(is.na(heart))

set.seed(123)

## ----------------------------------------------------------------------------------------------------------------------------------
summary(heart)


## ----------------------------------------------------------------------------------------------------------------------------------
colnames(heart)[3] <- "CPK"
colnames(heart)[5] <- "eject_frac"
colnames(heart)[6] <- "high_bp"
colnames(heart)[7] <- "plat"
colnames(heart)[8] <- "serum_c"
colnames(heart)[9] <- "serum_s"
colnames(heart)[11] <- "smoke"


## ----------------------------------------------------------------------------------------------------------------------------------
#target variable distribution
ggplot(heart, aes(x=factor(fatal_mi))) + 
  geom_bar(fill="pink") + 
  labs(x = "Death Event", y = "Count")


## ----------------------------------------------------------------------------------------------------------------------------------
DataExplorer::plot_histogram(heart, ncol = 3)


## ----------------------------------------------------------------------------------------------------------------------------------
ggplot(heart,
       aes(x = age, y = fatal_mi)) +
  geom_point()


## ----------------------------------------------------------------------------------------------------------------------------------
DataExplorer::plot_boxplot(heart, by = "fatal_mi", ncol = 3)


## ----------------------------------------------------------------------------------------------------------------------------------
corr_mat <- round(cor(heart), 2)
corr_mat

melted_corr <- melt(corr_mat)

ggplot(data = melted_corr, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0, limit=c(-1,1), space="Lab", name="Correlation") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(title="Feature Correlation Heatmap")



## ----------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

#Split into training
heart_split <- initial_split(heart)
heart_train <- training(heart_split)

# Splitting the training into validate and test
heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)


## ----------------------------------------------------------------------------------------------------------------------------------
#Normalizing the data
cake <- recipe(fatal_mi ~ ., data = heart) |>
  step_impute_mean(all_numeric())|>
  step_center(all_numeric(), -anaemia, -diabetes, -high_bp, -sex, -smoke, -fatal_mi) |>
  step_scale(all_numeric(), -anaemia, -diabetes, -high_bp, -sex, -smoke, -fatal_mi) |>
  step_unknown(all_nominal(), -all_outcomes()) |>
  step_dummy(all_nominal(), one_hot = TRUE) |>
  prep(training = heart_train)

#Applying the preprocessing
heart_train <- bake(cake, new_data = heart_train)
heart_validate <- bake(cake, new_data = heart_validate)
heart_test <- bake(cake, new_data = heart_test)


## ----------------------------------------------------------------------------------------------------------------------------------
model.lr <- glm(fatal_mi~., data=heart_train, family=binomial)

predict.lr <- predict(model.lr, heart_test, type="response")
predict.lr <- ifelse(predict.lr > 0.5, 1, 0)


## ----------------------------------------------------------------------------------------------------------------------------------
model.dt <- rpart(fatal_mi~., data=heart_train, method="class")

predict.dt <- predict(model.dt, heart_test, type="prob")
predict.dt <- ifelse(predict.dt[,2] > 0.5, 1, 0)  # Convert probabilities to binary class labels


## ----------------------------------------------------------------------------------------------------------------------------------
model.rf <- randomForest(fatal_mi~., data=heart_train, ntree=500)

predict.rf <- predict(model.rf, heart_test, type="response")
predict.rf <- ifelse(predict.rf > 0.5, 1, 0)


## ----------------------------------------------------------------------------------------------------------------------------------
model.svm <- svm(fatal_mi~., data=heart_train, probability=TRUE, kernel="linear")

predict.svm <- predict(model.svm, heart_test)
predict.svm <- ifelse(predict.svm > 0.5, 1, 0)


## ----------------------------------------------------------------------------------------------------------------------------------
conf.mat.lr.test <- confusionMatrix(as.factor(predict.lr), as.factor(heart_test$fatal_mi))
conf.mat.lr.test

conf.mat.dt.test <- confusionMatrix(as.factor(predict.dt), as.factor(heart_test$fatal_mi))
conf.mat.dt.test

conf.mat.rf.test <- confusionMatrix(as.factor(predict.rf), as.factor(heart_test$fatal_mi))
conf.mat.rf.test

conf.mat.svm.test <- confusionMatrix(as.factor(predict.svm), as.factor(heart_test$fatal_mi))
conf.mat.svm.test


## ----------------------------------------------------------------------------------------------------------------------------------
roc.lr <- roc(heart_test$fatal_mi, as.numeric(predict.lr))
roc.dt <- roc(heart_test$fatal_mi, as.numeric(predict.dt))
roc.rf <- roc(heart_test$fatal_mi, as.numeric(predict.rf))
roc.svm <- roc(heart_test$fatal_mi, as.numeric(predict.svm))

plot(roc.lr, col="blue", main="ROC", print.auc=TRUE)
lines(roc.rf, col="red")
lines(roc.svm, col="green")
lines(roc.dt, col="purple")
legend("bottomright", legend=c("Logistic", "Random Forest", "SVM", "Decision Tree"), col=c("blue", "red", "green", "purple"), lwd=2)



## ----------------------------------------------------------------------------------------------------------------------------------
cat("Random Forest AUC:", auc(roc.rf), "\n")
cat("Logistic Regression AUC:", auc(roc.lr), "\n")
cat("SVM AUC:", auc(roc.svm), "\n")
cat("Decision Tree AUC:", auc(roc.dt), "\n")


## ----------------------------------------------------------------------------------------------------------------------------------
heart_train_cv <- heart_train
heart_test_cv <- heart_test

heart_train_cv$fatal_mi <- factor(heart_train_cv$fatal_mi, levels = c(0, 1), labels = c("No", "Yes"))
heart_test_cv$fatal_mi <- factor(heart_test_cv$fatal_mi, levels = c(0, 1), labels = c("No", "Yes"))

# Cross-validation setup
control <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

# Hyperparameter tuning for Decision Tree with Cross-validation
model.dt.cv <- train(fatal_mi~., data=heart_train_cv, method="rpart", trControl=control, tuneLength=10, metric="ROC")
predict.dt.cv <- predict(model.dt.cv, heart_test_cv)
roc.dt.cv <- roc(heart_test_cv$fatal_mi, as.numeric(predict.dt.cv))

# Hyperparameter tuning for Random Forest with Cross-validation
model.rf.cv <- train(fatal_mi~., data=heart_train_cv, method="rf", trControl=control, tuneLength=10, metric="ROC")
predict.rf.cv <- predict(model.rf.cv, heart_test_cv)
roc.rf.cv <- roc(heart_test_cv$fatal_mi, as.numeric(predict.rf.cv))

# Hyperparameter tuning for SVM with Cross-validation
model.svm.cv <- train(fatal_mi~., data=heart_train_cv, method="svmRadial", trControl=control, tuneLength=10, metric="ROC")
predict.svm.cv <- predict(model.svm.cv, heart_test_cv)
roc.svm.cv <- roc(heart_test_cv$fatal_mi, as.numeric(predict.svm.cv))

# Hyperparameter tuning for Logistic Regression with Cross-validation
model.lr.cv <- train(fatal_mi~., data=heart_train_cv, method="glm", family="binomial", trControl=control, metric="ROC")
predict.lr.cv <- predict(model.lr.cv, heart_test_cv, type="prob")
roc.lr.cv <- roc(heart_test_cv$fatal_mi, predict.lr.cv[,2])

# ROC
plot(roc.lr.cv, col="blue", main="ROC Improved Models", print.auc=TRUE)
lines(roc.rf.cv, col="red")
lines(roc.svm.cv, col="green")
lines(roc.dt.cv, col="purple")
legend("bottomright", legend=c("Logistic Regression (CV)", "Random Forest (CV)", "SVM (CV)", "Decision Tree (CV)"), col=c("blue", "red", "green", "purple"), lwd=2)

cat("Random Forest (CV) AUC:", auc(roc.rf.cv), "\n")
cat("Logistic Regression (CV) AUC:", auc(roc.lr.cv), "\n")
cat("SVM (CV) AUC:", auc(roc.svm.cv), "\n")
cat("Decision Tree (CV) AUC:", auc(roc.dt.cv), "\n")


## ----------------------------------------------------------------------------------------------------------------------------------
final.pred <- predict(model.rf, heart_validate)

final.pred <- ifelse(final.pred > 0.5, 1, 0)


## ----------------------------------------------------------------------------------------------------------------------------------
conf.mat.final <- confusionMatrix(as.factor(final.pred), as.factor(heart_validate$fatal_mi))
conf.mat.final


## ----------------------------------------------------------------------------------------------------------------------------------
heart_validate_cv <- heart_validate

heart_validate_cv$fatal_mi <- factor(heart_validate_cv$fatal_mi, levels = c(0, 1), labels = c("No", "Yes"))

final.pred.cv <- predict(model.rf.cv, heart_validate)

roc.rf.cv <- roc(heart_test_cv$fatal_mi, as.numeric(predict.rf.cv))

cat("Random Forest (CV) AUC for Validation Set:", auc(roc.rf.cv), "\n")

final.pred.cv


## ----------------------------------------------------------------------------------------------------------------------------------
conf.mat.final.cv <- confusionMatrix(as.factor(final.pred.cv), as.factor(heart_validate_cv$fatal_mi))
conf.mat.final.cv

