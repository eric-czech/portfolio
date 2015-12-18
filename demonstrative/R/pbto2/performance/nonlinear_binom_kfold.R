library(foreach)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(rstan)
library(reshape2)
library(cvTools)
library(doMC)
library(ROCR)
library(loo)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_binom_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_utils.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

# Set timeseries feature to be used in modeling
ts.feature <- c('pao2')

static.features <- c('age', 'marshall', 'gcs', 'sex')
features <- c(static.features, ts.feature)

# Load pre-aggregated timeseries dataset
dwu <- get.wide.data(scale.vars=F, outcome.func=gos.to.binom, reset.uid=F) %>% mutate(uid=as.integer(uid))
dwu <- dwu %>% select(starts_with(ts.feature), one_of(static.features), one_of(c('gos', 'uid')))
dws <- dwu %>% mutate_each(funs(scale), -uid, -gos)
dwf <- dws %>% select(-uid, -gos) %>% names

# Load non-aggregated timeseries dataset
dlu <- get.long.data(features, scale.vars=F, outcome.func=gos.to.binom, reset.uid=F, rm.na=T)
dls <- dlu %>% mutate_each_(funs(scale), features)

# Determine intersection of uids for both datasets above
# * This is only necessary because NA's in non-aggregated data result 
#   in 0's in aggregated timeseries features (so removing by NA only will not work)
uids <- sort(intersect(unique(dws$uid), unique(dls$uid))) 
rm.uids <- setdiff(unique(c(dws$uid, dls$uid)), uids) %>% paste(., collapse=',') 
print(paste0('Dropping uids: ', rm.uids))

dwu <- filter(dwu, uid %in% uids)
dws <- filter(dws, uid %in% uids)
dlu <- filter(dlu, uid %in% uids)
dls <- filter(dls, uid %in% uids)

set.seed(123)
k <- 10
folds <- cvFolds(length(uids), K = k, type = 'random')

setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
dlogit.model <- 'nonlinear_binom_kfold.stan'
slogit.model <- 'nonlinear_slogit_binom_kfold.stan'
linear.model <- 'logistic_regression_kfold.stan'

reset.uid <- function(d) d %>% mutate(uid=as.integer(factor(uid)))

run.fold <- function(fold, ...){
  print(paste0('Running fold number ', fold))
  uid.tr <- uids[folds$subsets[folds$which != fold]]
  uid.ho <- uids[folds$subsets[folds$which == fold]]

  # Create long dataset for modeling
  dl.all <- dls %>% reset.uid
  dl.tr <- dls %>% filter(uid %in% uid.tr) %>% reset.uid
  dl.ho <- dls %>% filter(uid %in% uid.ho) %>% reset.uid
  uid.nl <- head(unique(dl.tr$uid), 2)
  
  dl.stan.cv <- get.stan.data.cv(dl.tr, dl.ho, static.features, ts.feature)
  dl.stan.fl <- get.stan.data.cv(dl.all, dl.ho %>% filter(uid %in% uid.nl), static.features, ts.feature)
  
  # Create wide dataset for modeling
  dw.all <- dws %>% select(-uid)
  dw.tr <- dws %>% filter(uid %in% uid.tr) %>% select(-uid)
  dw.ho <- dws %>% filter(uid %in% uid.ho) %>% select(-uid)
  
  dw.stan.cv <- get.wide.model.data(dw.tr, dwf, d.ho=dw.ho)
  dw.stan.fl <- get.wide.model.data(dw.all, dwf, d.ho=head(dw.ho, 2))

  null.feat <- c('gos', static.features)
  dn.stan.cv <- get.wide.model.data(dw.tr[,null.feat], static.features, d.ho=dw.ho[,null.feat])
  dn.stan.fl <- get.wide.model.data(dw.all[,null.feat], static.features, d.ho=head(dw.ho[,null.feat], 2))
  
  #browser()
  # Run nonlinear, double logistic, unaggregated timeseries model
  ml.cv <- stan(dlogit.model, data = dl.stan.cv, ...)
  if (fold == 1)
    ml.fl <- stan(dlogit.model, data = dl.stan.fl, ...)
  else
    ml.fl <- NULL
  
  # Run nonlinear, single logistic, unaggregated timeseries model
  ms.cv <- stan(slogit.model, data = dl.stan.cv, ...)
  if (fold == 1)
    ms.fl <- stan(slogit.model, data = dl.stan.fl, ...)
  else
    ms.fl <- NULL
  
  # Run linear, aggregated timeseries model
  mw.cv <- stan(linear.model, data = dw.stan.cv, ...)
  if (fold == 1)
    mw.fl <- stan(linear.model, data = dw.stan.fl, ...)
  else
    mw.fl <- NULL
  
  # Run linear, null model
  mn.cv <- stan(linear.model, data = dn.stan.cv, ...)
  if (fold == 1)
    mn.fl <- stan(linear.model, data = dn.stan.fl, ...)
  else
    mn.fl <- NULL
  
  list(
    fold=fold,
    dl=dl.stan.cv, ml.cv=ml.cv, ml.fl=ml.fl, # Double logistic results
    ds=dl.stan.cv, ms.cv=ms.cv, ms.fl=ms.fl, # Single logistic results
    dw=dw.stan.cv, mw.cv=mw.cv, mw.fl=mw.fl, # Aggregated linear results
    dn=dn.stan.cv, mn.cv=mn.cv, mn.fl=mn.fl  # Null results
  )
}


# res <- foreach(i=1:k) %do% run.fold(i, warmup = 100, iter = 1000, thin = 3, chains = 1, verbose = FALSE)

registerDoMC(6) # Only run this once to avoid crashes
res <- foreach(i=1:k) %dopar% run.fold(i, warmup = 100, iter = 1000, thin = 3, 
                                              chains = 1, verbose = FALSE)

res.file <- sprintf('/Users/eczech/data/pbto2/cache/res_%s.Rdata', ts.feature)
save(res, file=res.file)

#posteriors <- foreach(i=1:k) %do% run.fold(i)

# pars <- c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha', 'p')
# m <- res[[1]]$mn.cv
# rstan::extract(m)
# summary(m)$summary[,'Rhat']


rhat <- extract.rhat(res)
plot.rhat(rhat)

waic <- extract.waic(res)
waic

preds <- extract.predictions(res)

# compute.fold.logloss <- function(preds){
#   preds %>% group_by(model, fold) %>% do({
#     
#     p <- prediction(.$y.pred, .$y.true)
#     auc <- p %>% performance('auc') %>% .@y.values
#     data.frame(auc=auc[[1]])
#   })
# }
compute.fold.auc <- function(preds){
  preds %>% group_by(model, fold) %>% do({
    p <- prediction(.$y.pred, .$y.true)
    auc <- p %>% performance('auc') %>% .@y.values
    data.frame(auc=auc[[1]])
  })
}
auc <- compute.fold.auc(preds)
auc %>% ggplot(aes(x=model, y=auc, color=model)) + geom_jitter(position = position_jitter(width = .1))
auc %>% group_by(model) %>% summarise(mean(auc), sd(auc))


# Future possibilities
## Comparing WAIC differences
# compare(waic(log_lik1), waic(log_lik2))

## ROC Curves
# prediction(y$y_pred_glm, y$y_true) %>% performance('tpr', 'fpr') %>% plot

