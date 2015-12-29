library(glmulti)
library(dplyr)
library(cvTools)
library(MASS)
library(MLmetrics)
library(dummies)

select <- dplyr::select
source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_utils.R')

d <- get.wide.data(outcome.func=gos.to.binom)
do <- get.wide.data(outcome.func=gos.to.ord) %>% mutate(gos=factor(gos))


m <- glm(gos ~ age + sex + gcs + marshall + pbto2_100_inf + pbto2_0_20 + pao2_0_30 + pao2_100_inf, data=d, family='binomial')
m <- glm(gos ~ age + sex + gcs + marshall + pbto2_0_20 + pao2_0_30, data=d, family='binomial')

# 
# m <- polr(gos ~ age + sex + gcs + marshall + pbto2_100_inf + pbto2_0_20 + pao2_0_30 + pao2_100_inf, data=do, Hess=T)
# coefs <- coef(summary(m))
# bc <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
# coefs <- cbind(coefs, "p value" = bc)
# coefs

p <- c('age', 'sex', 'marshall', 'gcs')
#p <- c('1')

get.form <- function(e) as.formula(paste0('gos ~ ', paste(c(p, e), collapse=' + ')))
# models <- list(
#   all=get.form(c('icp1_20_inf', 'paco2_0_35', 'paco2_45_inf')),
#   lower_only=get.form(c('paco2_0_35')),
#   upper_only=get.form(c('paco2_45_inf', 'icp1_20_inf')),
#   paco2_both=get.form(c('paco2_0_35', 'paco2_45_inf')),
#   pao2_upper=get.form(c('paco2_45_inf')),
#   icp_only=get.form(c('icp1_20_inf')),
#   none=get.form(c())
# )
models <- list(
  all=get.form(c('pbto2_0_20', 'pbto2_100_inf', 'pao2_0_30', 'pao2_100_inf', 'icp1_20_inf', 'paco2_45_inf', 'paco2_0_35')),
  lower_only=get.form(c('pbto2_0_20', 'pao2_0_30', 'paco2_0_35')),
  upper_only=get.form(c('pbto2_100_inf', 'pao2_100_inf', 'paco2_45_inf')),
  icp_paco2=get.form(c('icp1_20_inf', 'paco2_45_inf', 'paco2_0_35')),
  pbto2_pao2=get.form(c('pbto2_0_20', 'pbto2_100_inf', 'pao2_0_30', 'pao2_100_inf')),
  pao2=get.form(c('pao2_100_inf', 'pao2_0_30')),
  pbto2=get.form(c('pbto2_0_20', 'pbto2_100_inf')),
  icp=get.form(c('icp1_20_inf')),
  paco2=get.form(c('paco2_45_inf', 'paco2_0_35')),
  none=get.form(c())
)

ic.glm.binary <- function(f, score=AIC){
  r <- glm(f, data=d, family='binomial')
  #browser()
  score(r)
}

bin.predict.class <- function(f, d.tr, d.ho){
  r <- glm(f, data=d.tr, family='binomial')
  r <- predict(r, newdata=d.ho, type='response')
  ifelse(r > .5, 1, 0)
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
accuracy.score <- function(y.true, y.pred){
  sum(y.true == y.pred) / length(y.true)
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
  #browser()
  eps <- 1e-15
  N <- nrow(y_pred)
  y_pred <- pmax(pmin(y_pred, 1 - eps), eps)
  mll <- (-1/N) * sum(y_true * log(y_pred))
  return(mll)
}
cv.run <- function(f, d, predictor, score){
  set.seed(123)
  k <- nrow(d) 
  #k <- 50
  folds <- cvFolds(nrow(d), K = k, type = 'random')
  fold.res <- foreach(i=1:k, .combine=rbind) %do% {
    d.tr <- d[folds$subsets[folds$which != i],]
    d.ho <- d[folds$subsets[folds$which == i],]
    y <- predictor(f, d.tr, d.ho)
    data.frame(score=score(d.ho$gos, y))
  } 
  fold.res$score %>% mean
}


scores <- foreach(m=names(models), .combine=rbind)%do%{
  #score <- ic.glm.binary(models[[m]], score=AIC)
  #score <- ic.glm.binary(models[[m]], score=AICc)
  #score <- ic.glm.binary(models[[m]], score=BIC)
  #score <- cv.run(models[[m]], d, predictor=bin.predict.class, score=accuracy.score)
  #score <- cv.run(models[[m]], d, predictor=bin.predict.probs, score=logloss)
  #score <- cv.run(models[[m]], do, predictor=ord.predict.probs, score=mlogloss)
  score <- cv.run(models[[m]], do, predictor=ord.predict.class, score=accuracy.score)
  data.frame(model=m, formula=deparse(models[[m]], width.cutoff = 500), score=score)
} %>% arrange(score)
scores

summary(scores$score)
paste(mean(scores$score), '+/-', sd(scores$score))
