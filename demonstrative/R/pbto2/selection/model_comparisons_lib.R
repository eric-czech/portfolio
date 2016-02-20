library(AICcmodavg)
library(ROCR)
library(caTools)
library(gbm)
library(randomForest)
library(ggplot2)
library(caret)
library(foreach)
library(iterators)
library(caret)

get.form <- function(e) as.formula(paste0('gos ~ ', paste(e, collapse=' + ')))

scale.df <- function(d) d %>% mutate_each(funs(scale), -gos, -uid)

prep.df <- function(d, ts.features, scale.vars=T){
  if (length(ts.features) > 0){
    ts.na <- d %>% select_(.dots=paste0(ts.features, '_is_na')) %>% apply(1, sum)
    d <- d %>% filter(ts.na == 0)
  }
  if (scale.vars) {
    d %>% select(-ends_with('_is_na')) %>% scale.df
  } else {
    d %>% select(-ends_with('_is_na'))
  }
}

run.model <- function(m, d, modelfun, cv.score=score.predictions, 
                      ic.score=NULL, prep.with.all.vars=F, cv.folds=NULL){
  set.seed(1)
  
  # Determine unique predictor prefixes in this model
  if (prep.with.all.vars) m.prefix <- all.vars
  else m.prefix <- m[!m %in% p]
  m.prefix <- str_split(m.prefix, '_') %>% sapply(function(x)x[1]) %>% unique
  d <- prep.df(d, m.prefix)
  
  #browser()
  # Create model formula
  form <- get.form(m)
  
  # Run LOO CV loop and compute scores
  if (is.null(folds))
    folds <- createFolds(1:nrow(d), 1:nrow(d), returnTrain=T)
  preds <- foreach(fold=folds, i=icount(), .combine=rbind) %dopar% {
    d.tr <- d[fold,]
    d.ho <- d[-fold,]
    modelfun(form, d.tr, d.ho) %>% 
      mutate(y.true=d.ho$gos, i=i)
  } 
  cv.scores <- cv.score(preds) %>% mutate(n=nrow(d))
  
  # Compute IC scores, if possible
  if (!is.null(ic.score)) cv.scores <- cbind(cv.scores, ic.score(form, d))
  
  list(cv.scores=cv.scores, preds=preds, data=d)
}

run.models <- function(modelfun, prep.with.all.vars, ic.score=NULL, model.filter=c(), cv.folds=NULL){
  res <- foreach(m=names(models))%do%{
    if (length(model.filter) > 0 && !m %in% model.filter)
      return(NULL)
    #score <- cv.run(models[[m]], d, predictor=bin.predict.class, score=accuracy.score)
    #score <- cv.run(models[[m]], d, predictor=bin.predict.probs, score=logloss)
    #score <- cv.run(models[[m]], do, predictor=ord.predict.probs, score=mlogloss)
    #score <- cv.run(models[[m]], dbu, predictor=ord.predict.class, score=accuracy.score)
    r <- run.model(models[[m]], dbu, modelfun=modelfun, ic.score=ic.score, 
                   prep.with.all.vars=prep.with.all.vars, cv.folds=cv.folds) 
    r[['cv.scores']] <- r[['cv.scores']] %>% mutate(model=m, formula=paste(models[[m]], collapse=' + '))
    r
  } 
  res[!sapply(res, is.null)] 
}

get.cv.scores <- function(res) foreach(r=res, .combine=rbind) %do% r$cv.scores

get.cv.results <- function(r, s, desc=T){
  r <- get.cv.scores(r)
  r[order(r[,s], decreasing = desc),] %>% select(-formula)
}

get.models <- function(p){
  gas.models <- list(
    icp=c('icp1_20_inf'),
    paco2=c('paco2_0_28', 'paco2_42_inf'),
    pao2=c('pao2_0_300', 'pao2_875_inf'),
    pbto2=c('pbto2_0_20', 'pbto2_70_inf'),
    pao2_pbto2=c('pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf'),
    icp_paco2=c('icp1_20_inf', 'paco2_0_28', 'paco2_42_inf'),
    icp_pao2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf'),
    icp_pbto2=c('icp1_20_inf', 'pbto2_0_20', 'pbto2_70_inf'),
    icp_pao2_pbto2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf'),
    icp_pao2_pbto2_paco2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf', 'paco2_0_28', 'paco2_42_inf')
  )
  all.vars <- unlist(gas.models) %>% unique 
  models1 <- gas.models
  models1[['demo']] <- c('age', 'sex')
  models2 <- lapply(gas.models, function(x) c(p, x))
  names(models2) <- sapply(names(models2), function(x)paste0('wcov_', x))
  models2[['wcov_none']] <- p
  models <- c(models1, models2)
  list(models=models, all.vars=all.vars)
}

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

pred.binary.svm <- function(form, d.tr, d.ho){
  ctrl <- trainControl(method = "cv", number=5, classProbs=TRUE)
  d.svm <- d.tr %>% mutate(gos=factor(gos, levels=c(0, 1), labels=c('bad', 'good')))
  r <- train(form, data=d.svm, method = "svmRadial", trControl = ctrl)
  
  p <- predict(r, newdata=d.ho, type='prob')[1,2]
  y <- predict(r, newdata=d.ho, type='raw') %>% as.integer - 1
  data.frame(y.pred=y, y.proba=p)
}

pred.binary.knn <- function(form, d.tr, d.ho){
  ctrl <- trainControl(method="cv", number=5)
  d.knn <- d.tr %>% mutate(gos=factor(gos, levels=c(0, 1), labels=c('bad', 'good')))
  r <- train(form, data = d.knn, method = "knn", trControl = ctrl, tuneLength = 20)
  p <- predict(r, newdata=d.ho, type='prob')[1,2]
  y <- predict(r, newdata=d.ho, type='raw') %>% as.integer - 1
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

plot.roc.curve <- function(res, model.filter, title='ROC Curves'){
  roc <- foreach(r=res, .combine=rbind) %do% {
    model <- r$cv.scores$model[1]
    if (!model %in% model.filter)
      return(NULL)
    pred.obj <- prediction(r$preds$y.proba, r$preds$y.true)
    roc <- pred.obj %>% performance('tpr', 'fpr') 
    auc <- pred.obj %>% performance('auc') %>% .@y.values %>% .[[1]]
    data.frame(model=model, auc=auc, x=roc@x.values[[1]], y=roc@y.values[[1]])
  }
  roc %>% 
    mutate(model=paste0(model, ' (AUC=', round(auc, 2), ')')) %>%
    ggplot(aes(x=x, y=y, color=model)) + geom_line() + 
    geom_abline(intercept = 0, slope = 1) + 
    theme_bw() + ggtitle(title)
}

get.model.diffs <- function(res, model1, model2){
  probs <- foreach(r=res, .combine=rbind) %do% {
    model <- r$cv.scores$model[1]
    if (!model %in% c(model1, model2))
      return(NULL)
    data.frame(y.proba=r$preds$y.proba, y.true=r$preds$y.true, i=r$preds$i, model=model, stringsAsFactors=F)
  } 
  probs %>% dcast(y.true + i ~ model, value.var = 'y.proba') %>% 
    mutate(lik1=y.true * .[,model1] + (1-y.true) * (1-.[,model1])) %>% 
    mutate(lik2=y.true * .[,model2] + (1-y.true) * (1-.[,model2])) %>% 
    mutate(llr=log(lik1 / lik2))
}
