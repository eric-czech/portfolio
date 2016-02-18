library(MASS); library(plyr); library(dplyr)
library(caret)
library(AppliedPredictiveModeling)
source('~/repos/portfolio/functional/ml/R/trainer.R')
source('~/repos/portfolio/functional/ml/R/results.R')
library(devtools)
library(ROCR)

# unload(inst("caretEnsemble")); install_local('/Users/eczech/repos/misc/caretEnsemble'); library(caretEnsemble)
library(caretEnsemble)

library(doMC)
registerDoMC(5)
select <- dplyr::select

SEED <- 123

d <- new.env()
data(concrete, envir=d)
d <- d$concrete %>% 
  mutate(response = ifelse(CompressiveStrength > mean(CompressiveStrength), 'pos', 'neg')) %>%
  mutate(response = factor(response, levels=c('neg', 'pos'))) %>%
  select(-CompressiveStrength)

set.seed(SEED)
split.index <- createDataPartition(y = d$response, p = .75, list = FALSE)
d.train <- d[split.index,]
d.test  <- d[-split.index,]
fold.index <- createFolds(d.train$response, k=10)

sum.fun <- function(...) c(twoClassSummary(...), defaultSummary(...))
trControl <- trainControl(
  method="cv",
  number=10,
  savePredictions="final",
  classProbs=TRUE,
  index=fold.index,
  summaryFunction=sum.fun
)


model.list <- caretList(
  response ~ ., data=d.train,
  trControl=trControl,
  methodList=c("glmnet", "rpart", "rda")
)
m <- caretStack(model.list, method='rf', trControl=trainControl(number=8, classProbs=TRUE, savePredictions='final'))
y.pred <- predict(m, newdata=d.test %>% dplyr::select(-response), type='raw')
y.prob <- predict(m, newdata=d.test %>% dplyr::select(-response), type='prob')
preds <- data.frame(y.pred=y.pred, y.prob=y.prob, y.test=d.test[,'response'])
table(preds$y.pred, preds$y.test)

# This computes the pearson correlation between the selected
# performance measure for each model (e.g. Accuracy).  Note that
# this is not the correlation in the predictions themselves
modelCor(resamples(model.list))


# Ensembling with trainer

basicConfig(level=loglevels['DEBUG'])
trctrl <- function(index, ...) trainControl(
  method="cv", number=10, savePredictions="final", classProbs=T, 
  index=index, summaryFunction=sum.fun, returnData=T, ...)
predict.class <- function(fit, d, i){ list(
  prob=predict(fit, d$X.test[,names(d$X.train)], type='prob')[,2], 
  class=predict(fit, d$X.test[,names(d$X.train)], type='raw')
)}
test.selector <- function(d) d$y.test

X <- d.train %>% select(-response); y <- d.train$response

trainer <- Trainer(cache.dir='/tmp/ensemble_test', cache.project='iris', seed=SEED)
index.gen <- function(y, index, level){
  if (level == 1) createFolds(y, k = 10, returnTrain = T)
  else if (level == 2) createFolds(y[index], k=8, returnTrain = T)
  else stop(sprintf('Folds at level %s not supported', level))
}
trainer$generateFoldIndex(y, index.gen)
fold.data.gen <- function(X.train, y.train, X.test, y.test){
  list(X.train=X.train, y.train=y.train, X.test=X.test, y.test=y.test)
}
trainer$generateFoldData(X, y, fold.data.gen, NULL, enable.cache=F)


models <- list()
GetRDAModel <- function(){
  m <- getModelInfo(model = "rda", regex = FALSE)[[1]]
  m$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {  
    klaR:::rda(x, y, gamma = param$gamma, lambda = param$lambda, ...)
  }
  m$method <- 'rda'
  m
}
GetKNNModel <- function(){
  m <- getModelInfo(model = "knn", regex = FALSE)[[1]]
  m$method <- 'knn'
  m
}
models$rda <- list(
  name='bin.rda', predict=predict.class, test=test.selector,
  train=function(d, idx, ...) train(d$X.train, d$y.train, method=GetRDAModel(),  tuneLength=3, trControl = trctrl(idx))
)
models$knn <- list(
    name='bin.knn', predict=predict.class, test=test.selector,
    train=function(d, idx, ...) train(d$X.train, d$y.train, method=GetKNNModel(),  tuneLength=3, trControl = trctrl(idx))
)
models$rf <- list(
  name='bin.rf', predict=predict.class, test=test.selector,
  train=function(d, idx, ...) train(d$X.train, d$y.train, method='rf',  tuneLength=3, trControl = trctrl(idx))
)
models$rpart <- list(
  name='bin.rpart', predict=predict.class, test=test.selector,
  train=function(d, idx, ...) train(d$X.train, d$y.train, method='rpart',  tuneLength=3, trControl = trctrl(idx))
)
models$gbm <- list(
  name='bin.gbm', predict=predict.class, test=test.selector,
  train=function(d, idx, ...) train(d$X.train, d$y.train, method='gbm',  tuneLength=3, trControl = trctrl(idx), verbose=F)
)

# Train base models
results <- list()

results$rda <- trainer$train(models$rda, enable.cache=F)
results$knn <- trainer$train(models$knn, enable.cache=F)
results$rf <- trainer$train(models$rf, enable.cache=F)
results$rpart <- trainer$train(models$rpart, enable.cache=F)
results$gbm <- trainer$train(models$gbm, enable.cache=F)

# Define ensemble model
ens.models <- list(
  knn=function(i) results$knn[[i]]$fit,
  rf=function(i) results$rf[[i]]$fit,
  rpart=function(i) results$rpart[[i]]$fit,
  gbm=function(i) results$gbm[[i]]$fit
)
models$ens <- list(
  name='ens', test=test.selector,
  train=function(d, idx, i, ...){
#     m <- lapply(ens.models, function(m) m(i))
#     class(m) <- "caretList"
    
    m <- caretList(
      d$X.train, d$y.train, trControl=trctrl(idx),
      methodList=c("knn", "rf", "rpart", "gbm"),
      tuneLength=3
    )
    x <- caretEnsemble(m, family='binomial', trControl=trainControl(number=1, classProbs=TRUE, savePredictions='final'))
    #x <- caretStack(m, method='glmnet', family='binomial')
    #x <- caretStack(m, method='rf', trControl=trainControl(number=8, classProbs=TRUE, savePredictions='final'))
    x
  }, predict=function(fit, d, i){ 
    list(
      prob=predict(fit, newdata=d$X.test, type='prob'),
      class=predict(fit, newdata=d$X.test, type='raw')
    )
  }
)
models$ens2 <- list(
  name='ens2', test=test.selector,
  train=function(d, idx, i, ...){
    m <- lapply(ens.models, function(m) m(i))
    class(m) <- "caretList"
    x <- caretEnsemble(m, family='binomial', trControl=trainControl(number=1, classProbs=TRUE, savePredictions='final'))
    x
  }, predict=function(fit, d, i){ 
    list(
      prob=predict(fit, newdata=d$X.test, type='prob'),
      class=predict(fit, newdata=d$X.test, type='raw')
    )
  }
)
results$ens <- trainer$train(models$ens, enable.cache=F)
results$ens2 <- trainer$train(models$ens2, enable.cache=F)


Summarize <- function(results){
  preds <- foreach(m=results, .combine=rbind) %do% {
    foreach(fold=m, .combine=rbind) %do% {
      #if (fold$model == 'ens') browser()
      acc <- sum(fold$y.pred$class == fold$y.test) / length(fold$y.test)
      pred <- prediction(fold$y.pred$prob, fold$y.test, label.ordering=c('neg', 'pos'))
      auc <- performance(pred, 'auc')
      data.frame(model=fold$model, fold=fold$fold, auc=auc@y.values[[1]], acc=acc)
    }
  }
}
cv.res <- Summarize(results)
cv.res %>% ggplot(aes(x=model, y=acc)) + geom_boxplot()
