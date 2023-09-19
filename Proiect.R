library(rpart)
library(tidyverse)
library(rsample)
library(caret)
library(tree)
library(ipred)
library(pROC)
library(corrplot)
library(modeldata)
library(rpart.plot)


###########################################################################################################################################

#                                                     PREGATIRE DATE


# incarcare date
titanic<- read_csv("titanic.csv")

# extragere denumire coloane - pas optional
names(titanic)

# vizualizare date
view(titanic)

# inlocuire date lipsa in Age cu valoarea medie rotunjita
titanic$Age[is.na(titanic$Age)] <- round(mean(titanic$Age, na.rm = TRUE))

# eliminare coloana Cabin
titanic <- select(titanic, -Cabin)

# inlcuire date lipsa cu NULL
titanic$Ticket <- NULL

# transformare 0 si 1 in YES si NO
titanic$Survived <- ifelse(titanic$Survived == 1, "Yes", "No")

# vizualizare date
view(titanic)


###########################################################################################################################################

#                                                     GRAFICE DE CORELATIE

titanic %>%
  select_if(is.numeric) %>%
  gather(metric,value) %>%
  ggplot(aes(value, fill=metric)) +
  geom_density(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free")

titanic <- titanic %>% 
  mutate(Embarked = factor(Embarked),
         Pclass = factor(Pclass),
         Survived = factor(Survived),
         Sex = factor(Sex))


# corelatie intre variabilele numerice
titanic %>%
  filter(Survived == "Yes") %>%
  select_if(is.numeric) %>%    
  cor() %>%
  corrplot::corrplot()


# corelatie intre variabilele numerice
titanic %>%
  filter(Survived == "No") %>%
  select_if(is.numeric) %>%    
  cor() %>%
  corrplot::corrplot()

###########################################################################################################################################

#                                                         IMPARTIRE DATE

set.seed(123)

# impartim tabelul in set de antrenament (70%) si set de validare (30%)
titanic_split <- initial_split(titanic, prop = 0.7, strata = "Survived")
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

# YES si NO - set intreg
table(titanic$Survived)
# YES si NO - 70%
table(titanic_train$Survived)
# YES si NO - 30%
table(titanic_test$Survived)

###########################################################################################################################################

#                                                         NAIVE BAYES

features <- setdiff(names(titanic_train), "Survived")
x <- titanic_train[,features]
y <- titanic_train$Survived

fitControl <- trainControl(
  method = "cv", #Cross Validation
  number = 10
)

modNbSimpleCV <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = fitControl
)
modNbSimpleCV
confusionMatrix(modNbSimpleCV)

searchGrid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0.5,
  adjust = seq(0, 5, by = 1)
)

modNbCVSearch <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = fitControl,
  tuneGrid = searchGrid
)

modNbCVSearch
confusionMatrix(modNbCVSearch)

modNbCVSearch$results %>%
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

pred <- predict(modNbCVSearch, titanic_test)
pred
predProb_naiveBayes <- predict(modNbCVSearch, titanic_test, type = "prob")
predProb_naiveBayes
confusionMatrix(pred, titanic_test$Survived)

###########################################################################################################################################

#                                                               ARBORI

# ARBORE 1

set.seed(123)
arbore1 = rpart(
  formula = Survived ~. ,
  data = titanic_train,
  method = "class"
)
arbore1
summary(arbore1)
rpart.plot(arbore1)
plotcp(arbore1)

#Predictie pe setul Test - folosind ARBORE arbore1
pred_arbore1 <- predict(arbore1, newdata = titanic_test, target = "class")
pred_arbore1 <- as_tibble(pred_arbore1) %>% mutate(class = ifelse(No >= Yes, "No", "Yes"))
pred_arbore1

table(pred_arbore1$class, titanic_test$Survived)
confusionMatrix(factor(pred_arbore1$class), factor(titanic_test$Survived))

# ARBORE 2
arbore2 = rpart(
  formula = Survived ~. ,
  data = titanic_train,
  method = "class",
  control = list(cp = 0)
)
arbore2
summary(arbore2)
rpart.plot(arbore2)
plotcp(arbore2)


#Predictie pe setul Test - folosind ARBORE arbore2
pred_arbore2 <- predict(arbore2, newdata = titanic_test, target = "class")
pred_arbore2 <- as_tibble(pred_arbore2) %>% mutate(class = ifelse(No >= Yes, "No", "Yes"))
pred_arbore2

table(pred_arbore2$class, titanic_test$Survived)
confusionMatrix(factor(pred_arbore2$class), factor(titanic_test$Survived))

#ARBORE 2 PRUNNED
set.seed(123)
arbore2_prunned <- prune(arbore2, cp=0,02)
arbore2_prunned

#Predictie pe setul Test - folosind ARBORE arbore2 PRUNNED
pred_arbore2_prunned <- predict(arbore2_prunned, newdata = titanic_test, target = "class")
pred_arbore2_prunned <- as_tibble(pred_arbore2_prunned) %>% mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_arbore2_prunned$class), factor(titanic_test$Survived))


set.seed(123)

#ENTROPIE

arbore1_tree <- tree(Survived ~., data = titanic_train) # works with deviance computed with entropy
arbore1_tree
summary(arbore1_tree)

pred_arbore1_tree <- predict(arbore1_tree, newdata = titanic_test, target = "class")
pred_arbore1_tree <- as_tibble(pred_arbore1_tree) %>% mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_arbore1_tree$class), factor(titanic_test$Survived))

#INDEX GINI

set.seed(123)
arbore1_tree_gini <- tree(Survived ~., data = titanic_train, split="gini") # works with Gini index
arbore1_tree_gini
summary(arbore1_tree_gini)

pred_arbore1_tree_gini <- predict(arbore1_tree_gini, newdata = titanic_test, target = "class")
pred_arbore1_tree_gini <- as_tibble(pred_arbore1_tree_gini) %>% mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_arbore1_tree_gini$class), factor(titanic_test$Survived))

###########################################################################################################################################

#                                                               BAGGING
set.seed(123)
bagged_m1 <- bagging(Survived ~ .,
                     data = titanic_train, 
                     coob = TRUE) # face validarea pe restul instantelor de dinafara bagului
bagged_m1
summary(bagged_m1)
pred_bagged_m1 <- predict(bagged_m1, newdata = titanic_test, target = "class")
confusionMatrix(pred_bagged_m1, factor(titanic_test$Survived))


ntree <- seq(10, 50, by = 1) # prima data face cu 10 bag, 11 bag, ... 50 bag
misclassification <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging( 
    Survived ~.,
    data = titanic_train,
    coob = TRUE,
    nbag = ntree[i])
  
  misclassification[i] = model$err #RMSE = eroarea medie
}

plot(ntree, misclassification, type="l", lwd="2")
axis(side = 1, at = seq(10, 50, by = 1)) 


bagged_m1_33 <- bagging(Survived ~ .,
                        data = titanic_train, coob = TRUE, nbag = 33)
bagged_m1_33
summary(bagged_m1_33)
pred_bagged_m1_33 <- predict(bagged_m1_33, newdata = titanic_test, target = "class")
confusionMatrix(pred_bagged_m1_33, factor(titanic_test$Survived))

###########################################################################################################################################

#                                                     COMPARARE REZULTATE - CURBA ROC

# NAIVE BAYES
dataset_pred_naiveBayes <- data.frame(
  actual.class = titanic_test$Survived,
  probability = predProb_naiveBayes[, 1]  
)  
roc.val_pred_naiveBayes <- roc(actual.class ~ probability, dataset_pred_naiveBayes)
adf_naiveBayes <- data.frame(
  specificity = roc.val_pred_naiveBayes$specificities,
  sensitivity = roc.val_pred_naiveBayes$sensitivities
)

# ARBORE 1
arbore1_roc <- predict(arbore1, titanic_test, type = "prob")
dataset_arbore1 <- data.frame(
  actual.class = titanic_test$Survived,
  probability = arbore1_roc[, 1]  
)  
roc.val_arbore1 <- roc(actual.class ~ probability, dataset_arbore1)
adf_arbore1 <- data.frame(
  specificity = roc.val_arbore1$specificities,
  sensitivity = roc.val_arbore1$sensitivities
)

# ARBORE2
arbore2_roc <- predict(arbore2, titanic_test, type = "prob")
dataset_arbore2 <- data.frame(
  actual.class = titanic_test$Survived,
  probability = arbore2_roc[, 1]  
)  
roc.val_arbore2 <- roc(actual.class ~ probability, dataset_arbore2)
adf_arbore2 <- data.frame(
  specificity = roc.val_arbore2$specificities,
  sensitivity = roc.val_arbore2$sensitivities
)

# ARBORE 2 PRUNNED
arbore2_prunned_roc <- predict(arbore2_prunned, titanic_test, type = "prob")
dataset_arbore2_prunned <- data.frame(
  actual.class = titanic_test$Survived,
  probability = arbore2_prunned_roc[, 1]  
)  
roc.val_arbore2_prunned <- roc(actual.class ~ probability, dataset_arbore2_prunned)
adf_arbore2_prunned <- data.frame(
  specificity = roc.val_arbore2_prunned$specificities,
  sensitivity = roc.val_arbore2_prunned$sensitivities
)

# ENTROPIE
arbore1_tree_roc <- predict(arbore1_tree, titanic_test)
dataset_arbore1_tree <- data.frame(
  actual.class = titanic_test$Survived,
  probability = arbore1_tree_roc[, 1]  
)  
roc.val_arbore1_tree <- roc(actual.class ~ probability, dataset_arbore1_tree)
adf_arbore1_tree <- data.frame(
  specificity = roc.val_arbore1_tree$specificities,
  sensitivity = roc.val_arbore1_tree$sensitivities
)

# INDEX GINI
arbore1_tree_gini_roc <- predict(arbore1_tree_gini, titanic_test)
dataset_arbore1_tree_gini <- data.frame(
  actual.class = titanic_test$Survived,
  probability = arbore1_tree_gini_roc[, 1]  
)  
roc.val_arbore1_tree_gini <- roc(actual.class ~ probability, dataset_arbore1_tree_gini)
adf_arbore1_tree_gini <- data.frame(
  specificity <- roc.val_arbore1_tree_gini$specificities,
  sensitivity <- roc.val_arbore1_tree_gini$sensitivities)

# BAGGING M1
bagged_m1_roc <- predict(bagged_m1, titanic_test, type = "prob")
dataset_bagged_m1 <- data.frame(
  actual.class = titanic_test$Survived,
  probability = bagged_m1_roc[, 1]  
)  
roc.val_bagged_m1 <- roc(actual.class ~ probability, dataset_bagged_m1)
adf_bagged_m1 <- data.frame(
  specificity = roc.val_bagged_m1$specificities,
  sensitivity = roc.val_bagged_m1$sensitivities
)

# BAGGING M1 33
bagged_m1_33_roc <- predict(bagged_m1_33, titanic_test, type = "prob")
dataset_bagged_m1_33 <- data.frame(
  actual.class = titanic_test$Survived,
  probability = bagged_m1_33_roc[, 1]  
)  
roc.val_bagged_m1_33 <- roc(actual.class ~ probability, dataset_bagged_m1_33)
adf_bagged_m1_33 <- data.frame(
  specificity = roc.val_bagged_m1_33$specificities,
  sensitivity = roc.val_bagged_m1_33$sensitivities
)

#REPREZENTARE GRAFIC
ggplot() +
  geom_line(data=adf_arbore1, aes(specificity,sensitivity), color='red') +
  geom_line(data=adf_arbore2, aes(specificity,sensitivity), color='orange') +
  geom_line(data=adf_arbore2_prunned, aes(specificity,sensitivity), color='green') +
  geom_line(data=adf_arbore1_tree, aes(specificity,sensitivity), color='blue') +
  geom_line(data=adf_arbore1_tree_gini, aes(specificity,sensitivity), color='purple') + 
  geom_line(data=adf_bagged_m1, aes(specificity,sensitivity), color='black') + 
  geom_line(data=adf_bagged_m1_33, aes(specificity,sensitivity), color='yellow') + 
  geom_line(data=adf_naiveBayes, aes(specificity,sensitivity), color='darkgrey') + 
  scale_x_reverse() +
  theme(text = element_text(size = 17))
