library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)
source('~/repos/portfolio/demonstrative/R/pbto2/common.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

d <- read.csv('~/data/pbto2/export/data_model_input_72hr_tsa.csv', stringsAsFactors=F)

features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')

d.stan <- d %>% 
  #sample.uids(frac=.75) %>%
  mutate(outcome=gos.3.binary) %>% 
  mutate_each_(funs(scale), features) %>%
  dplyr::select_(.dots=c(features, 'outcome', 'uid', 'tsa_min')) %>%
  mutate(uid=as.integer(factor(uid))) 


### Stan

d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid)

z.cp <- scale.var(seq(3, 30, length.out=28), d, 'pbto2')
t.cp <- seq(720, 4320, 240) # Every 4 hours between 12 hours and 72 hours



d.model <- list(
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  N_UID = max(d.stan$uid),
  N_Z_CP = length(z.cp),
  N_T_CP = length(t.cp),
  y = d.stan.uid %>% .$outcome %>% as.integer,
  x = d.stan.uid %>% dplyr::select(-outcome, -pbto2, -uid, -tsa_min),
  z = d.stan$pbto2,
  t = d.stan$tsa_min,
  uid = d.stan$uid,
  z_cutpoints = z.cp,
  t_cutpoints = t.cp
)


setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
#model.file <- 'cutpoint_rollup_binom_marginal.stan'
model.file <- 'time_rollup_binom.stan'


# posterior <- stan(model.file, data = d.model,
#                   warmup = 25, iter = 75, thin = 5, 
#                   chains = 1, verbose = FALSE)
posterior <- stan(model.file, data = d.model,
                  warmup = 50, iter = 100, thin = 5, 
                  chains = 1, verbose = FALSE)

# Running parallel chains on Mac
# library(parallel) # or some other parallelizing package
# n.chains <- 3
# 
# posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
#   stan(file = model.file, data = d.model, warmup = 10, iter = 60, chains = 1, thin = 5, 
#        verbose = FALSE, chain_id=chain)
# })
# posterior <- sflist2stanfit(posterior)



post <- rstan::extract(posterior)

rstan::traceplot(posterior, c('beta', 'beta_z_lo', 'beta_z_hi', 'z_cutpoint', 't_cutpoint'))
plot(posterior)

beta.post <- compute.var.posteriors(d, post)

plot.pbto2.cutoff(beta.post)

plot.time.cutoff(beta.post)

plot.beta.post(beta.post)

