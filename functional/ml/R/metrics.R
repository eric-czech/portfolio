library(dplyr)
library(ROCR)
EPSILON <- .000000000000001


.score.reg.mse <- function (y.true, y.pred) mean((y.true-y.pred)^2)
.score.reg.rmse <- function (y.true, y.pred) sqrt(.score.reg.mse(y.true, y.pred))
.score.reg.rsquared <- function(y.true, y.pred) cor(y.true, y.pred) ^ 2
.score.reg.mae <- function (y.true, y.pred) mean(abs(y.true-y.pred))

GetRegressorScore <- function(y.true, y.pred, score){
  if (!score %in% c('rsquared', 'rmse', 'mae', 'mse'))
    stop(sprintf('Score calculations for type "%s" not valid.', score))
  eval(parse(text = sprintf('.score.reg.%s(y.true, y.pred)', score)))
}

GetBinaryClassifierScore <- function(y.true, y.pred, score, is.proba=T){
  if (is.proba){
    if (!score %in% c('logloss', 'roc.auc', 'pr.auc'))
      stop(sprintf('Score calculations using probability predictions for score type "%s" not valid.', score))
    eval(parse(text = sprintf('.score.bin.%s(y.true, y.pred)', score)))
  } else {
    if (!score %in% c('accuracy'))
      stop(sprintf('Score calculations using class predictions for score type "%s" not valid.', score))
    eval(parse(text = sprintf('.score.bin.%s(y.true, y.pred)', score)))
  }
}

.score.bin.accuracy <- function(y.true, y.pred){
  sum(y.true == y.pred) / length(y.true)
}

.score.bin.logloss <- function(y.true, y.pred, epsilon=EPSILON) {
  yhat <- pmin(pmax(y.pred, epsilon), 1-epsilon)
  -mean(y.true*log(yhat) + (1-y.true)*log(1 - yhat))
}

.score.bin.roc.auc <- function(y.true, y.pred){
  prediction(y.pred, y.true) %>% performance('auc') %>% .@y.values %>% .[[1]]
}

.score.bin.pr.auc <- function(y.true, y.pred) {
  # Taken from https://github.com/andybega/auc-pr/blob/master/auc-pr.r
  perf <- performance(prediction(y.pred, y.true), 'prec', 'rec')
  xy <- data.frame(recall=perf@x.values[[1]], precision=perf@y.values[[1]])
  xy <- subset(xy, !is.nan(xy$precision))
  trapz(xy$recall, xy$precision)
}

GetMultinomialClassifierScore <- function(y.true, y.pred, score, is.proba=T){
  if (is.proba){
    if (!score %in% c('logloss'))
      stop(sprintf('Score calculations using probability predictions for score type "%s" not valid.', score))
    eval(parse(text = sprintf('.score.mc.%s(y.true, y.pred)', score)))
  } else {
    if (!score %in% c())
      stop(sprintf('Score calculations using class predictions for score type "%s" not valid.', score))
    eval(parse(text = sprintf('.score.mc.%s(y.true, y.pred)', score)))
  }
}
  
.score.mc.logloss <- function(y.true, y.pred, epsilon=EPSILON){
  #' Multiclass Log Loss score calculator
  #' 
  #' Args:
  #'    y.true: True class labels (can be factor or dummy encoded matrix)
  #'    y.pred: Predicted class labels (must be NxC matrix with class probabilities in each row)
  #' Returns:
  #'    Multi-class Log Loss score
  #' Example:
  #'    y.pred <- rbind(c(0.9,0.1,0), c(0.85,0.15,0))
  #'    y.true <- rbind(c(1,  0,  0), c(0,   1,   0))
  #'    .score.mc.logloss(y.true, y.pred) # = 1.00124
  #'    y.true <- factor(c(1, 2), levels=1:3)
  #'    .score.mc.logloss(y.true, y.pred) # = 1.00124
  if (is.vector(y.pred))
    y.pred <- y.pred %>% matrix %>% t
  if (is.factor(y.true)) {
    y.true.mat <- matrix(0, nrow = length(y.true), ncol = length(levels(y.true)))
    sample.levels <- as.integer(y.true)
    for (i in 1:length(y.true)) y.true.mat[i, sample.levels[i]] <- 1
    y.true <- y.true.mat
  }
  N <- nrow(y.pred)
  y.pred <- pmax(pmin(y.pred, 1 - epsilon), epsilon)
  (-1/N) * sum(y.true * log(y.pred))
}

