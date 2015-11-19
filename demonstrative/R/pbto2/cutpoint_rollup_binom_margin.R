library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

d <- read.csv('/Users/eczech/data/ptbo2/export/data_long_cutpoint_72hr.csv', stringsAsFactors=F)

features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')
scale <- function(x) (x - mean(x)) / sd(x)
d.stan <- d %>% 
  sample_frac(.05) %>% 
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
  pbto2 = d.stan$pbto2,
  uid = d.stan$uid,
  pbto2_cutpoints = scale.cutpoint(seq(1, 150, length.out=n.cp))
)

# Run the sampler
init_fun <- function() { list(
  beta=rep(0, d.model$N_VARS),
  beta_pbto2_lo=0,
  beta_pbto2_hi=0,
  alpha = 0
)} 
setwd('/Users/eczech/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'cutpoint_rollup_binom_marginal.stan'


# posterior <- stan(model.file, data = d.model,
#                   warmup = 10, iter = 50, thin = 1, 
#                   chains = 1, verbose = FALSE, init=init_fun)

library(parallel) # or some other parallelizing package
n.chains <- 3

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 10, iter = 60, chains = 1, thin = 5, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)
post <- rstan::extract(posterior)

rstan::traceplot(posterior, c('beta', 'beta_pbto2_lo', 'beta_pbto2_hi', 'pbto2_cutpoint_idx', 'pbto2_cutpoint'))
plot(posterior)



#exp(post$lp)


d.lp <- post$lp %>% melt(id.vars='rowname')
d.lp.iter <- d.lp %>% group_by(iterations) %>% summarise(min=min(value)) %>% 
  #filter(min > -800) %>% 
  .$iterations %>% unique
d.lp %>% 
  filter(iterations %in% d.lp.iter) %>%
  #ggplot(aes(x=factor(Var2), y=value)) + geom_boxplot()
  ggplot(aes(x=Var2, y=value, color=factor(iterations))) + geom_line()

unscale <- function(x, var) x * sd(d[,var]) + mean(d[,var])
beta.post <- data.frame(post$beta) %>% setNames(features[-1]) %>% dplyr::mutate(samp_id=1:nrow(.))
beta.post$pbto2_cp <- unscale(post$pbto2_cutpoint %>% as.numeric, 'pbto2')
beta.post$pbto2_hi <- post$beta_pbto2_hi %>% as.numeric
beta.post$pbto2_lo <- post$beta_pbto2_lo %>% as.numeric

p <- beta.post %>% 
  ggplot(aes(x=pbto2_cp)) + geom_histogram(binwidth=1, alpha=.5) + 
  theme_bw() + ggtitle('Pbto2 Cutpoint Estimates') + 
  xlab('Pbto2 Cutoff')
p
#p + ggsave('/Users/eczech/data/ptbo2/images/pbto2_cp1.png')

p <- beta.post %>% 
  dplyr::select(pbto2_lo, pbto2_hi, samp_id) %>%
  melt(id.vars='samp_id') %>%
  mutate(variable=ifelse(variable == 'pbto2_lo', 'Pbto2 Below Cutpoint', 'Pbto2 Above Cutpoint')) %>%
  ggplot(aes(x=value, fill=variable)) + geom_density(alpha=.5) +
  theme_bw() + xlab('Coefficient Value') + ylab('Density') +
  ggtitle('Coefficient 95% Intevals for Pbto2 Above and Below Cutpoint') 
p
#p + ggsave('/Users/eczech/data/ptbo2/images/pbto2_coef.png')

p <- beta.post %>% melt(id.vars='samp_id') %>% 
  filter(variable != 'pbto2_cp') %>%
  dplyr::group_by(variable) %>% 
  summarise(
    lo=quantile(value, .025), 
    mid=quantile(value, .5), 
    hi=quantile(value, .975)
  ) %>% dplyr::mutate(variable=factor(variable)) %>% 
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed') + 
  ggtitle('Coefficient 95% Intervals') + xlab('Coefficient Value') + ylab('Variable')
p
#p + ggsave('/Users/eczech/data/ptbo2/images/coefs1.png')


