

# random Forest
load('D:/Rwd/training_final2_scale')
library(randomForest)
training_final2_scale$PURCHASE_FLG = as.factor(training_final2_scale$PURCHASE_FLG)
data = training_final2_scale[sample(1:nrow(training_final2_scale), 20000), ]
ranFst = randomForest(PURCHASE_FLG~.,data = data, mtry = 4, importance = TRUE, ntree = 100)
error.ranFst = ranFst$err.rate[100, 1]
pred.rf = predict(ranFst, type = 'prob')
varImpPlot(ranFst)

#bagging
library('ipred')
bagging = bagging(PURCHASE_FLG~., data = data, nbagg = 100,coob = TRUE)
err.bag = bagging$err
err.bag

#Double Bagging
library('ipred')
library("MASS")
comb.lda = list(list(model = slda, predict = function(obj, newdata)
  predict(obj, newdata)$x))
doublebag = bagging(PURCHASE_FLG~., data = data, nbagg = 100, comb =
                      comb.lda)
f = function(formula, data){
  bagging(formula, data, nbagg = 100, comb = comb.lda)
}
error = errorest(PURCHASE_FLG~., data = data, model = f, estimator = 'cv')
error

#boosting
library(gbm)
boost=gbm(PURCHASE_FLG~.,data=data, distribution = 'adaboost', n.trees =
             1000, cv.folds = 10)
predclass = boost$cv.fitted
predclass[predclass > 0] = 0
predclass[predclass < 0] = 1
trueclass = data[, 1]
mse = mean(predclass != trueclass)

par(mfrow = c(1, 2))
best.iter <- gbm.perf(boost,method="cv")
print(best.iter)
##plot the performance based on estimated best number of trees
summary(boost,n.trees=best.iter)[1:5,]

##predict new obs based on best number of trees
#pred.boost=predict(boost,newdata=datavalid[,-1], best.iter, type =
#                     'response')
