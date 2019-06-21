library(MASS)
data(Boston)
str(Boston)
summary(Boston)


# splitting into testing and training data
set.seed(06119969)
index <- sample(nrow(Boston),nrow(Boston)*0.75)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]


# linear regression -------------------------------------------------------

model_lm <- lm(medv~. , data = boston.train)
summary(model_lm)
BIC(model_lm)
pred_lm <- predict(model_lm)
MSE_lm <- mean((pred_lm - boston.train$medv)^2)
  
### prediction

pred_test_lm <- predict (model_lm, boston.test)
MSPE_lm <- mean((pred_test_lm - boston.test$medv)^2)

#### variable selection
nullmodel=lm(medv~1, data=boston.train)
fullmodel=lm(medv~., data=boston.train)
# BIC stepwise selection
model_step <- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both', k=log(nrow(boston.train)))
summary(model_step)
BIC(model_step)
pred_step <- predict(model_step)
MSE_step <- mean((pred_step - boston.train$medv)^2)
## prediction
pred_step_test <- predict(model_step, boston.test)
MSPE_step <- mean((pred_step_test - boston.test$medv)^2)


# Regression Tree ---------------------------------------------------------

library(rpart)
library(rpart.plot)
par(mfrow =c(1,1))
boston.rpart <- rpart(formula = medv ~ ., data = boston.train)
boston.rpart
prp(boston.rpart,digits = 4, extra = 1)
#training error
boston.train.pred.tree = predict(boston.rpart)
MSE_regrtree <- mean((boston.train.pred.tree - boston.train$medv)^2)

### prediction

boston.test.pred.tree = predict(boston.rpart,boston.test)
MSPE_regtree <- mean((boston.test.pred.tree - boston.test$medv)^2)

### pruning
boston.largetree <- rpart(formula = medv ~ ., data = boston.train, cp = 0.001)
prp(boston.largetree)
plotcp(boston.largetree)
cp.prune <- boston.largetree$cptable[which.min(boston.largetree$cptable[,"xerror"]),"CP"]
pruned <- rpart(formula = medv ~., data = boston.train, cp = cp.prune) #enter pruned value here
prp(pruned)
printcp(boston.largetree)

prune_pred <- predict(pruned)
MSE_prune <- mean((prune_pred - boston.train$medv)^2)
# this tree is the same as the tree we had in the beginning. MSE is also the same. 



# Bagging -----------------------------------------------------------------

#install.packages("ipred")
library(ipred)
boston.bag<- bagging(medv~., data = boston.train, nbagg=100)
boston.bag
boston.train.pred <- predict(boston.bag)
MSE_bag <- mean((boston.train.pred - boston.train$medv)^2)

### prediction
boston.bag.pred<- predict(boston.bag, newdata = boston.test)
MSPE_bag <- mean((boston.test$medv - boston.bag.pred)^2)

ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag1<- bagging(medv~., data = boston.train, nbagg=ntree[i])
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.test$medv-boston.bag.pred1)^2)
}
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)


# Random Forest -----------------------------------------------------------
library(randomForest)
boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE)
boston.rf
boston.rf$importance
plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")
pred.rf <- predict(boston.rf)
MSE_rf <- mean((pred.rf - boston.train$medv)^2)

### prediction

boston.rf.pred<- predict(boston.rf, boston.test)
MSPE_rf <- mean((boston.test$medv-boston.rf.pred)^2)

## 

oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((boston.test$medv-predict(fit, boston.test))^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))

# Boosting ----------------------------------------------------------------

library(gbm)

boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)
par(mfrow=c(1,1))
plot(boston.boost, i="lstat")
plot(boston.boost, i="rm")
boston.boost.pred <- predict(boston.boost, n.trees = 10000)
MSE_boost <- mean((boston.boost.pred - boston.train$medv)^2)

### prediction
boston.boost.pred.test<- predict(boston.boost, boston.test, n.trees = 10000)
MSPE_boost <- mean((boston.test$medv-boston.boost.pred.test)^2)

ntree<- seq(100, 10000, 100)
predmat<- predict(boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)


# GAM ---------------------------------------------------------------------

#install.packages("mgcv")
library(mgcv)
## Create a formula for a model with a large number of variables:
gam_formula <- as.formula(paste("medv ~ s(crim) + s(zn) + s(indus) + (chas)+ s(nox) + s(rm) + s(age) + s(dis) + (rad) + s(tax) + s(ptratio) + s(black) +s(lstat)", collapse= "+"))
#+", paste(colnames(boston.train)[c(1:3,5:8,10:13)], collapse= "+")))

boston.gam <- gam(formula = gam_formula, family=gaussian , data = boston.train)

summary(boston.gam)
#none of the edfs are 1. so we dont remove anything from the spline

plot(boston.gam, shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)

AIC(boston.gam)
BIC(boston.gam)

# in sample prediction error
boston.gam.pred <- predict(boston.gam)
MSE_gam <- mean((boston.gam.pred - boston.train$medv)^2)

#out of sample prediction error
boston.gam.pred.test <- predict(boston.gam, boston.test)
MSPE_gam <- mean((boston.gam.pred.test - boston.test$medv)^2)

gam_formula1 <- as.formula(paste("medv ~ s(crim) + (zn) + s(indus) + (chas)+ s(nox) + s(rm) + s(age) + s(dis) + (rad) + s(tax) + s(ptratio) + s(black) +s(lstat)", collapse= "+"))
boston.gam1 <- gam(formula = gam_formula1, family=gaussian , data = boston.train)
summary(boston.gam1)
plot(boston.gam1, shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)


# neural network ----------------------------------------------------------

library(MASS)
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)

n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
plot(nn)

## in sample prediction
pr.nn1 <- compute(nn,train_[,1:13])

pr.nn_1 <- pr.nn1$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
train.r <- (train_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

# MSE of training set
MSE.nn1 <- sum((train.r - pr.nn_1)^2)/nrow(train_)
MSE.nn1


## oos prediction
pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
test.r <- (test_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

# MSE of testing set
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
MSE.nn

