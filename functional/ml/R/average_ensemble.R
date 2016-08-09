library(caret)
library(dplyr)

MOST_FREQUENT <- function(x, lvl) names(sort(table(x), decreasing=TRUE)[1])
CLASS_FREQUENCY <- function(x, lvl) sum(x == lvl[1])/length(x)
SPLIT_MEAN_PROB_ON_.5 <- function(x, lvl) ifelse(mean(x) < .5, lvl[2], lvl[1])
MEAN_PROB <- function(x, lvl) mean(x)

GetEnsembleAveragingModel <- function(
  class.to.class=function(x, lvl) names(sort(table(x), decreasing=TRUE)[1]), 
  class.to.prob=function(x, lvl) sum(x == lvl[1])/length(x), 
  prob.to.class=function(x, lvl) ifelse(mean(x) < .5, lvl[2], lvl[1]),
  prob.to.prob=function(x, lvl) mean(x)){
  list(
    label = "Averaging Model",
    library = NULL,
    loop = NULL,
    type = c("Classification"),
    parameters = data.frame(parameter = "parameter", class = "character", label = "parameter"),
    grid = function(x, y, len = NULL, search = "grid") data.frame(parameter="none"),
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
      list(lev=lev)
    },
    predict = function(modelFit, newdata, submodels = NULL) {
      all.factors <- all(sapply(newdata, is.factor)) || all(sapply(newdata, is.character))
      all.numeric <- all(sapply(newdata, is.numeric))
      if (!all.factors && !all.numeric) stop(paste(
        'Aggregating model only works if training data contains',
        'all factors OR all numeric values'
      ))
      
      if (all.factors) p <- apply(newdata, 1, class.to.class, modelFit$lev)
      else p <- apply(newdata, 1, prob.to.class, modelFit$lev)

      factor(p, levels=modelFit$lev)
    },
    prob = function(modelFit, newdata, submodels = NULL) {
      all.factors <- all(sapply(newdata, is.factor)) || all(sapply(newdata, is.character))
      all.numeric <- all(sapply(newdata, is.numeric))
      if (!all.factors && !all.numeric) stop(paste(
        'Aggregating model can only make probability predictions if',
        'new data contains all factors OR all numeric values'
      ))
      
      if (all.factors) p <- apply(newdata, 1, class.to.prob, modelFit$lev)
      else p <- apply(newdata, 1, prob.to.prob, modelFit$lev)
      
      p <- cbind(p, 1-p)
      dimnames(p)[[2]] <- modelFit$obsLevels
      p
    },
    varImp = NULL,
    predictors = function(x, ...) NULL,
    levels = function(x) if(any(names(x) == "obsLevels")) x$obsLevels else NULL,
    sort = NULL
  )
}

GetEnsembleQuantileModel <- function(){
  
  class.to.prob  <- function(x, lvl) sum(x == lvl[1])/length(x)
  class.to.class <- function(x, lvl) names(sort(table(x), decreasing=TRUE)[1])
  prob.to.class  <- function(x, lvl, p) ifelse(quantile(x, p) < .5, lvl[2], lvl[1])
  prob.to.prob   <- function(x, lvl, p) quantile(x, p)
  
  list(
    label = "Quantile Ensemble Model",
    library = NULL,
    loop = NULL,
    type = c("Classification"),
    parameters = data.frame(parameter = "quantile", class = "numeric", label = "Quantile"),
    grid = function(x, y, len = NULL, search = "grid") {
      if (search == "grid"){
        data.frame(quantile=seq(0, 1, len=len))
      } else {
        data.frame(quantile=runif(len))
      }
    },
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
      list(lev=lev, quantile=param$quantile)
    },
    predict = function(modelFit, newdata, submodels = NULL) {
      all.factors <- all(sapply(newdata, is.factor)) || all(sapply(newdata, is.character))
      all.numeric <- all(sapply(newdata, is.numeric))
      if (!all.factors && !all.numeric) stop(paste(
        'Aggregating model only works if training data contains',
        'all factors OR all numeric values'
      ))
      
      if (all.factors) p <- apply(newdata, 1, class.to.class, modelFit$lev)
      else p <- apply(newdata, 1, prob.to.class, modelFit$lev, modelFit$quantile)
      
      factor(p, levels=modelFit$lev)
    },
    prob = function(modelFit, newdata, submodels = NULL) {
      all.factors <- all(sapply(newdata, is.factor)) || all(sapply(newdata, is.character))
      all.numeric <- all(sapply(newdata, is.numeric))
      if (!all.factors && !all.numeric) stop(paste(
        'Aggregating model can only make probability predictions if',
        'new data contains all factors OR all numeric values'
      ))
      
      if (all.factors) p <- apply(newdata, 1, class.to.prob, modelFit$lev)
      else p <- apply(newdata, 1, prob.to.prob, modelFit$lev, modelFit$quantile)
      
      p <- cbind(p, 1-p)
      dimnames(p)[[2]] <- modelFit$obsLevels
      p
    },
    varImp = NULL,
    predictors = function(x, ...) NULL,
    levels = function(x) if(any(names(x) == "obsLevels")) x$obsLevels else NULL,
    sort = function(x) x[order(x$quantile),]
  )
}

GetCaretEnsembleModel <- function(caret.list.args, caret.stack.args){
  list(
    label = "Caret Ensemble Model",
    library = NULL,
    loop = NULL,
    type = c("Classification"),
    parameters = data.frame(parameter = "parameter", class = "character", label = "parameter"),
    grid = function(x, y, len = NULL, search = "grid") data.frame(parameter="none"),
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
      require(caretEnsemble)
      train.args <- c(list(x, y), caret.list.args)
      cl <- do.call('caretList', train.args)
      
      train.args <- caret.stack.args
      train.args$all.models <- cl
      do.call('caretStack', train.args)
    },
    predict = function(modelFit, newdata, submodels = NULL) {
      predict(modelFit, newdata=newdata, type='raw')
    },
    prob = function(modelFit, newdata, submodels = NULL) {
      predict(modelFit, newdata=newdata, type='prob')
    },
    varImp = NULL,
    predictors = function(x, ...) NULL,
    levels = function(x) if(any(names(x) == "obsLevels")) x$obsLevels else NULL,
    sort = NULL
  )
}

# Quick Test:
# n <- 100
# 
# X1 <- data.frame(x1=runif(n), x2=runif(n))
# X2 <- data.frame(x1=sample(c('pos', 'neg'), n, T), x2=sample(c('pos', 'neg'), n, T))
# y <- apply(X1, 1, function(x) if (mean(x) > .5 && x[1] > .2) 'pos' else 'neg')
# y <- factor(y, levels=c('pos', 'neg'))
# 
# m <- train(X1, y, method=GetEnsembleAveragingModel(), metric='Accuracy',
#       trControl=trainControl(method='cv', number=10, classProbs=T))
# m$resample
# 
# m <- train(X2, y, method=GetEnsembleAveragingModel(), metric='Accuracy',
#            trControl=trainControl(method='cv', number=10, classProbs=T))
# m$resample
# 
# m <- train(X2, y, method=GetEnsembleQuantileModel(), metric='Accuracy', tuneLength=5,
#            trControl=trainControl(method='cv', number=10, classProbs=T))
# 
# 
# 
# library(caretEnsemble)
# caret.list.args <- list(
#   trControl=trainControl(method='cv', number=3, classProbs=T, savePredictions='final'),
#   methodList=c("glm", "rpart"),
#   tuneList=list(
#     rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=2))
#   )
# )
# ens.model <- GetEnsembleAveragingModel()
# caret.stack.args <- list(method=ens.model, trControl=trainControl(method='none', classProbs=T))
# m <- train(
#   X1, y, method=GetCaretEnsembleModel(caret.list.args, caret.stack.args), 
#   trControl=trainControl(method='cv', number=5, classProbs=T)
# )

