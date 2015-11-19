library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)
library(parallel)

d <- read.csv('/Users/eczech/data/ptbo2/export/data_long_cutpoint_48hr.csv', stringsAsFactors=F)

features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')
group_gos <- function(x){
  if (x >= 4) 3
  else if (x >= 2) 2
  else 1
}
scale <- function(x) (x - mean(x)) / sd(x)
d.stan <- d %>% 
  mutate(outcome=sapply(gos.3, group_gos)) %>% 
  mutate_each_(funs(scale), features) %>%
  dplyr::select_(.dots=c(features, 'outcome')) 


n.cp <- 50

# Convert the raw data created into Stan data
scale.cutpoint <- function(x) (x - mean(d$pbto2)) / sd(d$pbto2)
d.model <- list(
  N_OUTCOME = length(unique(d.stan$outcome)), 
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  N_CP = n.cp,
  y = as.integer(d.stan$outcome),
  x = d.stan %>% dplyr::select(-outcome, -pbto2),
  pbto2 = d.stan$pbto2,
  pbto2_cutpoints = scale.cutpoint(seq(1, 50, length.out=n.cp))
)

#library(parallel) 
chains <- 4

setwd('/Users/eczech/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'cutpoint_long_ord_marginal.stan'

# Non-parallel fitting
posterior <- stan(file = model.file, data = d.model, warmup = 100, iter = 500, chains = 1, 
                  verbose = FALSE, chain_id=1)

# Parallel fitting
n.chains <- 3
posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 250, iter = 1000, chains = 1, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)

save(posterior, file='/Users/eczech/data/ptbo2/export/mcmc_fit_long_cutpoint_v1.Rdata')
# 17448.1 seconds (Total)

rstan::traceplot(posterior, inc_warmup=F)
plot(posterior)

unscale <- function(x, var) x * sd(d[,var]) + mean(d[,var])


post <- rstan::extract(posterior)
#exp(post$lp)

d.lp <- post$lp %>% melt(id.vars='rowname')
d.lp.iter <- d.lp %>% group_by(iterations) %>% summarise(min=min(value)) %>% 
  #filter(min > -940) %>% 
  .$iterations %>% unique
d.lp %>% 
  filter(iterations %in% d.lp.iter) %>%
  ggplot(aes(x=factor(Var2), y=value)) + geom_boxplot()
  #ggplot(aes(x=Var2, y=value, color=factor(iterations))) + geom_line()

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


