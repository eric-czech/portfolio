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

source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_runner.R')
source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_utils.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

# Set timeseries feature to be used in modeling
ts.feature <- c('pha')

static.features <- c('age', 'marshall', 'gcs', 'sex')

# Run on mac for do par

# Parallel CV on Ubuntu (no dopar)
registerDoMC(15)
res <- run.cv(ts.feature, static.features, k=15, dopar=T, 
              warmup = 300, iter = 5300, thin = 5, chains = 1, verbose = FALSE)
# temp.res <- run.cv(ts.feature, static.features, k=2, dopar=F, 
#               warmup = 100, iter = 1000, thin = 3, chains = 1, verbose = FALSE)

# Parallel CV on Mac
# registerDoMC(6) # Only run this once to avoid crashes
# res <- run.cv(ts.feature, static.features, k=10, dopar=T, warmup = 25, iter = 50, thin = 2, chains = 1, verbose = FALSE)


res.file <- sprintf('~/data/pbto2/cache/cv_res_%s.Rdata', ts.feature)
save(res, file=res.file)

res.env <- new.env()
load(res.file, envir=res.env)
res <- res.env$res

#posteriors <- foreach(i=1:k) %do% run.fold(i)

# pars <- c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha', 'p')
# m <- res[[1]]$mn.cv
# rstan::extract(m)
# summary(m)$summary[,'Rhat']


rhat <- extract.rhat(res)
plot.rhat(rhat)


waic <- extract.waic(res)
waic 
waic %>% ggplot(aes(x=model, y=waic, ymin=waic-waic_se, ymax=waic+waic_se)) + geom_pointrange()

preds <- extract.predictions(res)

lloss <- compute.fold.logloss(preds)
lloss %>% ggplot(aes(x=model, y=logloss)) + geom_boxplot()

auc <- compute.fold.auc(preds)
auc %>% ggplot(aes(x=model, y=auc)) + geom_boxplot()
auc %>% ggplot(aes(x=model, y=auc, color=model)) + geom_jitter(position = position_jitter(width = .1))
auc %>% group_by(model) %>% summarise(mean(auc), sd(auc))


# Future possibilities
## Comparing WAIC differences
# compare(waic(log_lik1), waic(log_lik2))

## ROC Curves
# prediction(y$y_pred_glm, y$y_true) %>% performance('tpr', 'fpr') %>% plot

