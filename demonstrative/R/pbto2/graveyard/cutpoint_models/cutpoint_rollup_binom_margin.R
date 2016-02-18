library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

d <- read.csv('~/data/pbto2/export/data_model_input_72hr_tsa.csv', stringsAsFactors=F)

features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex', 'tsa_min')

d.stan <- d %>% 
  #sample.uids(frac=.75) %>%
  mutate(outcome=gos.3.binary) %>% 
  mutate_each_(funs(scale), features) %>%
  dplyr::select_(.dots=c(features, 'outcome', 'uid')) %>%
  mutate(uid=as.integer(factor(uid))) 


### Stan

d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid)
n.cp <- 150
scale.cutpoint <- function(x) (x - mean(d$pbto2)) / sd(d$pbto2)

d.model <- list(
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  N_UID = max(d.stan$uid),
  N_CP = n.cp,
  y = d.stan.uid %>% .$outcome %>% as.integer,
  x = d.stan.uid %>% dplyr::select(-outcome, -pbto2, -uid),
  z = d.stan$pbto2,
  uid = d.stan$uid,
  z_cutpoints = scale.cutpoint(seq(3, 30, length.out=n.cp))
)

# Run the sampler
init_fun <- function() { list(
  beta=rep(0, 4),
  beta_pbto2_lo=0,
  beta_pbto2_hi=0,
  alpha = 0
)} 
setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
#model.file <- 'cutpoint_rollup_binom_marginal_slow.stan'
model.file <- 'cutpoint_rollup_binom_marginal.stan'


# posterior <- stan(model.file, data = d.model,
#                   warmup = 25, iter = 75, thin = 5, 
#                   chains = 1, verbose = FALSE)
posterior <- stan(model.file, data = d.model,
                  warmup = 150, iter = 4000, thin = 5, 
                  chains = 14, verbose = FALSE)

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

rstan::traceplot(posterior, c('beta', 'beta_z_lo', 'beta_z_hi', 'z_cutpoint_idx', 'z_cutpoint'))
plot(posterior)



#exp(post$lp)

beta.post <- compute.var.posteriors(d, post)

plot.pbto2.cutoff(beta.post)

plot.beta.post(beta.post)

d.lp <- post$lp %>% melt(id.vars='rowname')
d.lp.iter <- d.lp %>% group_by(iterations) %>% summarise(min=min(value)) %>% 
  #filter(min > -800) %>% 
  .$iterations %>% unique
d.lp %>% 
  filter(iterations %in% d.lp.iter) %>%
  #ggplot(aes(x=factor(Var2), y=value)) + geom_boxplot()
  ggplot(aes(x=Var2, y=value, color=factor(iterations))) + geom_line()

beta.post %>% select(pbto2_cp, pbto2_lo) %>% 
  ggplot(aes(x=pbto2_cp, y=pbto2_lo)) + geom_point()
