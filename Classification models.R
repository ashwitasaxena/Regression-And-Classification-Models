# Downloading data and creating final dataset

library(caret)
german.data <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german.data) <- c("chk_acct", "duration", "credit_his", "purpose", 
                             "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                             "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                             "job", "n_people", "telephone", "foreign", "response")
german.data$response <- german.data$response- 1
german.data$response <- as.factor(german.data$response)
str(german.data)
summary(german.data)

# Split the Data into Training and Testing Set
set.seed(06119969)
trainrows <- sample(nrow(german.data), nrow(german.data) * 0.75)
germandata.train <- german.data[trainrows, ]
germandata.test <- german.data[-trainrows,]

####################################
# Multivariate Logistic Regression #
####################################

germandata.train.glm0 <- glm(response~., family = binomial, germandata.train)
summary(germandata.train.glm0)
modela <- step(germandata.train.glm0)
summary(modela)
germandata.train.glm0<-modela
summary(germandata.train.glm0)

pred.glm.gtrain.glm0 <- predict(germandata.train.glm0, type = "response")
pred.glm.gtest.glm0 <- predict(germandata.train.glm0, newdata=germandata.test,type = "response")
#residual deviance
germandata.train.glm0$deviance

# Asymmetric Misclassification Rate, using  5:1 asymmetric cost
cost <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

pcut <-  1/6 ## Bayes estimate

#training data stats
  class.pred.train.glm0<- (pred.glm.gtrain.glm0>pcut)*1
  table(germandata.train$response, class.pred.train.glm0, dnn = c("True", "Predicted"))
#  MR.glm0 <- mean(germandata.train$response!=class.pred.train.glm0)
  cost.train <- cost(r = germandata.train$response, pi = class.pred.train.glm0) 
#testing data stats
  class.pred.test.glm0<- (pred.glm.gtest.glm0>pcut)*1
  table(germandata.test$response, class.pred.test.glm0, dnn = c("True", "Predicted"))
#  MR.glm0.test <- mean(germandata.test$response!=class.pred.test.glm0)
  cost.test <- cost(r = germandata.test$response, pi = class.pred.test.glm0) 

#area under the curve
library(verification)
  par(mfrow=c(1,1))
roc.logit <- roc.plot(x=(germandata.train$response == "1"), pred =pred.glm.gtrain.glm0)
roc.logit$roc.vol
roc.logit.test <- roc.plot(x=(germandata.test$response == "1"), pred =pred.glm.gtest.glm0)
roc.logit.test$roc.vol

#######################
# Classification Tree #
#######################

library(rpart)
germandata.largetree <- rpart(formula = response~., data = germandata.train, 
                              parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)), cp=0.001)
library(rpart.plot)
prp(germandata.largetree, extra = 1, nn.font=40,box.palette = "pink")

plotcp(germandata.largetree)

printcp(germandata.largetree)

#Pruning##
german.prunedtree <- rpart(response~., data = germandata.train, method = "class",
                     parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)),cp=0.007533)
prp(german.prunedtree, extra = 1, nn.font=500,box.palette = "orange")


pred.tree.gtrain <- predict(german.prunedtree, type = "prob")
pred.tree.gtest <- predict(german.prunedtree, newdata=germandata.test, type = "prob")

#training stats
german.train.pred.rpart = as.numeric(pred.tree.gtrain[,2] > pcut)
table(germandata.train$response, german.train.pred.rpart, dnn=c("Truth","Predicted"))
#MR.rpart <- mean(germandata.train$response!=german.train.pred.rpart)
cost(germandata.train$response,german.train.pred.rpart)
#testing stats
german.test.pred.rpart = as.numeric(pred.tree.gtest[,2] > pcut)
table(germandata.test$response, german.test.pred.rpart, dnn=c("Truth","Predicted"))
#MR.rpart.test <- mean(germandata.test$response!=german.test.pred.rpart)
cost(germandata.test$response,german.test.pred.rpart)
#area under the curve
library(verification)
par(mfrow=c(1,2))
roc.tree <- roc.plot(x=(germandata.train$response == "1"), pred =pred.tree.gtrain[,2])
roc.tree$roc.vol

roc.tree.test <- roc.plot(x=(germandata.test$response == "1"), pred =pred.tree.gtest[,2])
roc.tree.test$roc.vol

###############################
# GAM (Genral Additive Model) #
###############################

library(mgcv)
str(germandata.train)
germandata.gam <- gam(as.factor(response)~chk_acct+s(duration)+credit_his+purpose+s(amount)+saving_acct+present_emp+installment_rate+sex+other_debtor+present_resid+property
                  +s(age)+other_install+housing+n_credits+telephone+foreign , family=binomial,data=germandata.train)
par(mfrow=c(1,2))
summary(germandata.gam)
plot(germandata.gam, shade=TRUE)

# Move age to partially linear term and refit gam() model
germandata.gam <- gam(as.factor(response)~chk_acct+s(duration)+credit_his+purpose+s(amount)+saving_acct+present_emp+installment_rate+sex+other_debtor+present_resid+property
                      +(age)+other_install+housing+n_credits+telephone+foreign , family=binomial,data=germandata.train)

summary(germandata.gam)
plot(germandata.gam, shade=TRUE)

#In sample performance
prob.gam.in<-predict(germandata.gam,germandata.train,type="response")
pred.gam.in<-(prob.gam.in>=pcut)*1
table(germandata.train$response,pred.gam.in,dnn=c("Observed","Predicted"))
#mean(ifelse(germandata.train$response != pred.gam.in, 1, 0))
cost(germandata.train$response, pred.gam.in)

#Out-of-sample performance########
prob.gam.out<-predict(germandata.gam,germandata.test,type="response")
pred.gam.out<-(prob.gam.out>=pcut)*1
table(germandata.test$response,pred.gam.out,dnn=c("Observed","Predicted"))
#mean(ifelse(germandata.test$response != pred.gam.out, 1, 0))
cost(germandata.test$response, pred.gam.out)

###ROC curve for GAM ##############
par(mfrow=(c(1,2)))
roc.gam <- roc.plot(x=(germandata.train$response == "1"), pred =prob.gam.in)
roc.gam$roc.vol

roc.gam.test <- roc.plot(x=(germandata.test$response == "1"), pred =prob.gam.out)
roc.gam.test$roc.vol

##############
# Neural net #
##############

library(nnet)
library(NeuralNetTools)
#library(neuralnet)
#library(dplyr)
library(e1071)
par(mfrow=c(1,1))
germandata.nnet <- train(response~., data=germandata.train,method="nnet")
print(germandata.nnet)
plot(germandata.nnet)
plotnet(germandata.nnet$finalModel, y_names = "response")
title("Graphical Representation of our Neural Network")

#In sample
prob.nnet= predict(germandata.nnet,type='prob')
pred.nnet = (prob.nnet[,2] >=pcut)*1
table(germandata.train$response,pred.nnet, dnn=c("Observed","Predicted"))
#mean(ifelse(germandata.train$response != pred.nnet, 1, 0))
cost(germandata.train$response, pred.nnet)

#Out of sample
prob.nnet.test = predict(germandata.nnet,germandata.test,type='prob')
pred.nnet.test = as.numeric(prob.nnet.test[,2] > pcut)
table(germandata.test$response,pred.nnet.test, dnn=c("Observed","Predicted"))
#mean(ifelse(germandata.test$response != pred.nnet.test, 1, 0))
cost(germandata.test$response, pred.nnet.test)


#Roc curve for nnet#
par(mfrow=c(1,2))
roc.nnet <- roc.plot(x=(germandata.train$response == "1"), pred =prob.nnet[,2])
roc.nnet$roc.vol

roc.nnet.test <- roc.plot(x=(germandata.test$response == "1"), pred =prob.nnet.test[,2])
roc.nnet.test$roc.vol







