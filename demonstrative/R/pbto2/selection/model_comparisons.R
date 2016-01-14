library(glmulti)
library(dplyr)
library(cvTools)
library(MASS)
library(MLmetrics)
library(dummies)
library(foreach)
library(plotly)

select <- dplyr::select
source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/selection/model_comparisons_lib.R')


dbu <- get.wide.data(outcome.func=gos.to.binom, scale.vars=F, remove.na.flags=F)
dou <- get.wide.data(outcome.func=gos.to.ord, scale.vars=F, remove.na.flags=F) %>% mutate(gos=factor(gos))
p <- c('age', 'sex', 'marshall', 'gcs')
#p <- c('sex')

m <- glm(gos ~ age + sex + gcs + marshall + pbto2_0_20 + pao2_0_30, data=d, family='binomial')

# 
# m <- polr(gos ~ age + sex + gcs + marshall + pbto2_100_inf + pbto2_0_20 + pao2_0_30 + pao2_100_inf, data=do, Hess=T)
# coefs <- coef(summary(m))
# bc <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
# coefs <- cbind(coefs, "p value" = bc)
# coefs

scale.df <- function(d) d %>% mutate_each(funs(scale), -gos, -uid)

prep.df <- function(d, ts.features){
  if (length(ts.features) > 0){
    ts.na <- d %>% select_(.dots=paste0(ts.features, '_is_na')) %>% apply(1, sum)
    d <- d %>% filter(ts.na == 0)
  }
  d %>% select(-ends_with('_is_na')) %>% scale.df
}

#get.form <- function(e) as.formula(paste0('gos ~ ', paste(c(p, e), collapse=' + ')))
get.form <- function(e) as.formula(paste0('gos ~ ', paste(e, collapse=' + ')))

gas.models <- list(
  icp=c('icp1_20_inf'),
  paco2=c('paco2_42_inf'),
  pao2=c('pao2_0_300', 'pao2_875_inf'),
  pbto2=c('pbto2_0_20', 'pbto2_70_inf'),
  pao2_pbto2=c('pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf'),
  icp_paco2=c('icp1_20_inf', 'paco2_42_inf'),
  icp_pao2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf'),
  icp_pbto2=c('icp1_20_inf', 'pbto2_0_20', 'pbto2_70_inf'),
  icp_pao2_pbto2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf'),
  icp_pao2_pbto2_paco2=c('icp1_20_inf', 'pao2_0_300', 'pao2_875_inf', 'pbto2_0_20', 'pbto2_70_inf', 'paco2_42_inf')
#   L_pao2=c('pao2_0_300'),
#   L_pbto2=c('pbto2_0_20'),
#   L_pao2_pbto2=c('pao2_0_300', 'pbto2_0_20'),
#   L_icp_paco2=c('icp1_20_inf', 'paco2_0_28'),
#   L_icp_pao2=c('icp1_20_inf', 'pao2_0_300'),
#   L_icp_pbto2=c('icp1_20_inf', 'pbto2_0_20'),
#   L_icp_pao2_pbto2=c('icp1_20_inf', 'pao2_0_300', 'pbto2_0_20'),
#   U_pao2=c('pao2_875_inf'),
#   U_pbto2=c('pbto2_70_inf'),
#   U_pao2_pbto2=c('pao2_875_inf', 'pbto2_70_inf'),
#   U_icp_paco2=c('icp1_20_inf', 'paco2_42_inf'),
#   U_icp_pao2=c('icp1_20_inf', 'pao2_875_inf'),
#   U_icp_pbto2=c('icp1_20_inf', 'pbto2_70_inf'),
#   U_icp_pao2_pbto2=c('icp1_20_inf', 'pao2_875_inf', 'pbto2_70_inf')
)

all.vars <- unlist(gas.models) %>% unique 
models1 <- gas.models
models1[['demo']] <- c('age', 'sex')
models2 <- lapply(gas.models, function(x) c(p, x))
names(models2) <- sapply(names(models2), function(x)paste0('wcov_', x))
models2[['wcov_none']] <- p
models <- c(models1, models2)

run.model <- function(m, d, modelfun, cv.score=score.predictions, ic.score=NULL, prep.with.all.vars=F){
  set.seed(1)
  
  # Determine unique predictor prefixes in this model
  if (prep.with.all.vars) m.prefix <- all.vars
  else m.prefix <- m[!m %in% p]
  m.prefix <- str_split(m.prefix, '_') %>% sapply(function(x)x[1]) %>% unique
  d <- prep.df(d, m.prefix)
  
  #browser()
  # Create model formula
  form <- get.form(m)
  #print(form)
  # Run LOO CV loop and compute scores
  preds <- foreach(i=1:nrow(d), .combine=rbind) %do% {
    d.tr <- d[-i,]
    d.ho <- d[i,]
    modelfun(form, d.tr, d.ho) %>% 
      mutate(y.true=d.ho$gos, i=i)
  } 
  cv.scores <- cv.score(preds) %>% mutate(n=nrow(d))
  
  # Compute IC scores, if possible
  if (!is.null(ic.score)) cv.scores <- cbind(cv.scores, ic.score(form, d))
  
  list(cv.scores=cv.scores, preds=preds)
}

run.models <- function(modelfun, prep.with.all.vars, ic.score=NULL){
  foreach(m=names(models))%do%{
    #score <- cv.run(models[[m]], d, predictor=bin.predict.class, score=accuracy.score)
    #score <- cv.run(models[[m]], d, predictor=bin.predict.probs, score=logloss)
    #score <- cv.run(models[[m]], do, predictor=ord.predict.probs, score=mlogloss)
    #score <- cv.run(models[[m]], dbu, predictor=ord.predict.class, score=accuracy.score)
    r <- run.model(models[[m]], dbu, modelfun=modelfun, ic.score=ic.score, prep.with.all.vars=prep.with.all.vars) 
    r[['cv.scores']] <- r[['cv.scores']] %>% mutate(model=m, formula=paste(models[[m]], collapse=' + '))
    r
  } 
}


results.glm <- run.models(pred.binary.glm, T, ic.score=ic.binary.glm)
results.gbm <- run.models(pred.binary.gbm, T)
results.rf <- run.models(pred.binary.rf, T)

get.cv.scores <- function(res) foreach(r=res, .combine=rbind) %do% r$cv.scores

get.cv.results <- function(r, s, desc=T){
  r <- get.cv.scores(r)
  r[order(r[,s], decreasing = desc),] %>% select(-formula)
}

r <- results.glm
cv.res <- get.cv.results(r, 'auc', desc=F)
cv.res <- get.cv.results(r, 'aucpr', desc=F)
cv.res <- get.cv.results(r, 'aicc')
cv.res <- get.cv.results(r, 'aic')
model.filter <- c(
  'demo', 'wcov_none', 'wcov_icp', 'wcov_icp_pao2', 'wcov_icp_paco2',
  'wcov_icp_pbto2', 'wcov_icp_pao2_pbto2', 'wcov_icp_pao2_pbto2_paco2'
)
plot.roc.curve(r, model.filter) %>% ggplotly %>% layout(showlegend = T)




m <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2')) %>% 
  gbm(get.form(models[['wcov_pbto2']]), data = ., n.trees = 1000, distribution='bernoulli',
         interaction.depth = 5, shrinkage=.1, n.minobsinnode = 20)
plot(m, i.var = 5, lwd = 2, col = "blue", main = "")

m <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2')) %>% 
  glm(get.form(models[['wcov_icp_pao2_pbto2_paco2']]), data=., family='binomial')
m <- dbu %>% prep.df(c('pbto2')) %>% 
  glm(get.form(models[['wcov_pbto2']]), data=., family='binomial')

library(glmulti)
d.glmulti <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2'))
m <- glmulti(get.form(models[['wcov_icp_pao2_pbto2_paco2']]), data=d.glmulti, family='binomial', level=1)
summary(m@objects[[1]])

d.glmulti <- dbu %>% prep.df(c('pbto2', 'pao2'))
m <- glmulti(get.form(models[['wcov_pao2_pbto2']]), data=d.glmulti, family='binomial', level=1)
summary(m@objects[[1]])
