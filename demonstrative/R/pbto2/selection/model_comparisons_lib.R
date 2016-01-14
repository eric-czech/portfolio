library(AICcmodavg)
library(ROCR)
library(caTools)
library(gbm)
library(randomForest)
library(ggplot2)

ic.binary.glm <- function(form, d){
  r <- glm(form, data=d, family='binomial')
  data.frame(aic=AIC(r), bic=BIC(r), aicc=AICc(r))
}

pred.binary.glm <- function(form, d.tr, d.ho){
  r <- glm(form, data=d.tr, family='binomial')
  p <- predict(r, newdata=d.ho, type='response')
  y <- ifelse(p > .5, 1, 0)
  data.frame(y.pred=y, y.proba=p)
}


pred.binary.gbm <- function(form, d.tr, d.ho){
  r <- gbm(form, data = d.tr, n.trees = 1000, distribution='bernoulli',
           interaction.depth = 5, shrinkage=.1, n.minobsinnode = 20)
  p <- predict(r, newdata=d.ho, type='response',  n.trees = 100)
  y <- ifelse(p > .5, 1, 0)
  data.frame(y.pred=y, y.proba=p)
}

pred.binary.rf <- function(form, d.tr, d.ho){
  r <- randomForest(form, data = d.tr %>% mutate(gos=factor(gos)))
  p <- predict(r, newdata=d.ho, type='prob')[1,2]
  y <- predict(r, newdata=d.ho, type='response') %>% as.character %>% as.integer
  data.frame(y.pred=y, y.proba=p)
}

pred.binary.gbm <- function(form, d.tr, d.ho){
  r <- gbm(form, data = d.tr, n.trees = 1000, distribution='bernoulli',
           interaction.depth = 5, shrinkage=.1, n.minobsinnode = 20)
  p <- predict(r, newdata=d.ho, type='response',  n.trees = 100)
  y <- ifelse(p > .5, 1, 0)
  data.frame(y.pred=y, y.proba=p)
}

bin.predict.probs <- function(f, d.tr, d.ho){
  r <- glm(f, data=d.tr, family='binomial')
  predict(r, newdata=d.ho, type='response')
}

ord.predict.class <- function(f, d.tr, d.ho){
  r <- polr(f, data=d.tr, Hess=TRUE)
  predict(r, newdata=d.ho, type='class')
}

ord.predict.probs <- function(f, d.tr, d.ho){
  r <- polr(f, data=d.tr, Hess=TRUE)
  predict(r, newdata=d.ho, type='probs')
}

score.predictions <- function(preds){
  data.frame(
    acc=score.accuracy(preds$y.true, preds$y.pred),
    lloss=score.logloss(preds$y.true, preds$y.proba),
    auc=score.auc(preds$y.true, preds$y.proba),
    aucpr=score.auc.pr(preds$y.true, preds$y.proba),
    tp=sum(preds$y.pred == 1 & preds$y.true == 1),
    fp=sum(preds$y.pred == 1 & preds$y.true == 0),
    tn=sum(preds$y.pred == 0 & preds$y.true == 0),
    fn=sum(preds$y.pred == 0 & preds$y.true == 1)
  )
}

score.accuracy <- function(y.true, y.pred){
  sum(y.true == y.pred) / length(y.true)
}

score.logloss <- function(y.true, y.pred, epsilon=.000000000000001) {
  yhat <- pmin(pmax(y.pred, epsilon), 1-epsilon)
  -mean(y.true*log(yhat) + (1-y.true)*log(1 - yhat))
}

score.auc <- function(y.true, y.pred){
  prediction(y.pred, y.true) %>% performance('auc') %>% .@y.values %>% .[[1]]
}

score.auc.pr <- function(y.true, y.pred) {
  # Taken from https://github.com/andybega/auc-pr/blob/master/auc-pr.r
  xx.df <- prediction(y.pred, y.true)
  perf  <- performance(xx.df, "prec", "rec")
  xy    <- data.frame(recall=perf@x.values[[1]], precision=perf@y.values[[1]])
  xy <- subset(xy, !is.nan(xy$precision))
  res   <- trapz(xy$recall, xy$precision)
  res
}

mlogloss <- function(y_true, y_pred){
  if (is.vector(y_pred))
    y_pred <- y_pred %>% matrix %>% t
  if (is.factor(y_true)) {
    y_true_mat <- matrix(0, nrow = length(y_true), ncol = length(levels(y_true)))
    sample_levels <- as.integer(y_true)
    for (i in 1:length(y_true)) y_true_mat[i, sample_levels[i]] <- 1
    y_true <- y_true_mat
  }
  eps <- 1e-15
  N <- nrow(y_pred)
  y_pred <- pmax(pmin(y_pred, 1 - eps), eps)
  mll <- (-1/N) * sum(y_true * log(y_pred))
  return(mll)
}

plot.roc.curve <- function(res, model.filter){
  roc <- foreach(r=res, .combine=rbind) %do% {
    model <- r$cv.scores$model[1]
    if (!model %in% model.filter)
      return(NULL)
    roc <- prediction(r$preds$y.proba, r$preds$y.true) %>%
      performance('tpr', 'fpr') 
    data.frame(model=model, x=roc@x.values[[1]], y=roc@y.values[[1]])
  }
  roc %>% ggplot(aes(x=x, y=y, color=model)) + geom_line() + 
    geom_abline(intercept = 0, slope = 1) + 
    theme_bw()
}

