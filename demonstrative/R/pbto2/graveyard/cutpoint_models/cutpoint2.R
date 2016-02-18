library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

d <- read.csv('/Users/eczech/data/ptbo2/export/data_long_cutpoint.csv', stringsAsFactors=F)


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
  dplyr::select_(.dots=c(features, 'outcome', 'uid')) %>%
  mutate(uid=as.integer(factor(uid)))


### Stan

stan.model <- '
data {
  int<lower=2> N_OUTCOME;                  // # of possible GOS outcomes 
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  int<lower=2> N_UID;
  int<lower=1> uid[N_OBS];
  real pbto2[N_OBS]; 

  int<lower=1, upper=N_OUTCOME> y[N_UID];  // Vector of outcomes
  row_vector[N_VARS] x[N_UID];           // Matrix of covariates (eg age, marshall, gcs)

}
parameters {
  vector[N_VARS] beta;   
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
  real beta_pbto2_lo;
  real beta_pbto2_hi;
  real<lower=-10,upper=10> pbto2_cutpoint;
}
model {
  vector[N_UID] x_above;
  vector[N_UID] x_below;
  for (i in 1:N_UID){
    x_above[i] <- 0;
    x_below[i] <- 0;
  }

  for (i in 1:N_OBS){
    if (pbto2[i] > pbto2_cutpoint)
      x_above[uid[i]] <- x_above[uid[i]] + 1;
    else
      x_below[uid[i]] <- x_below[uid[i]] + 1;
  }
  for (i in 1:N_UID){
    x_above[i] <- x_above[i];
    x_below[i] <- x_below[i];
  }

  beta ~ normal(0, 5);
  pbto2_cutpoint ~ normal(0, 3);
  beta_pbto2_lo ~ normal(0, 5);
  beta_pbto2_hi ~ normal(0, 5);
  
  for (i in 1:N_UID){
    y[i] ~ ordered_logistic(x[i] * beta + beta_pbto2_lo * x_below[i] + beta_pbto2_hi * x_above[i], outcome_cutpoints);
  }
}
'

d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid)

# Convert the raw data created into Stan data
d.model <- list(
  N_OUTCOME = length(unique(d.stan$outcome)), 
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  N_UID = max(d.stan$uid),
  y = d.stan.uid %>% .$outcome %>% as.integer,
  x = d.stan.uid %>% dplyr::select(-outcome, -pbto2, -uid),
  pbto2 = d.stan$pbto2,
  uid = d.stan$uid
)

# Run the sampler
init_fun <- function() { list(
  pbto2_cutpoint=0,
  beta=rep(0, d.model$N_VARS),
  beta_pbto2_lo=0,
  beta_pbto2_hi=0,
  outcome_cutpoints=c(-.5, .5)
)} 
fit <- stan(model_code = stan.model, data = d.model,
            warmup = 100, iter = 1000, thin = 25, chains = 1, verbose = FALSE, init=init_fun)

rstan::traceplot(fit)
plot(fit)


post <- rstan::extract(fit)
beta.post <- data.frame(post$beta) %>% setNames(features[-1]) %>% dplyr::mutate(samp_id=1:nrow(.))
beta.post[,'pbto2_lo'] <- post$beta_pbto2_lo %>% as.numeric
beta.post[,'pbto2_hi'] <- post$beta_pbto2_hi %>% as.numeric
beta.post[,'pbto2_cp'] <- post$pbto2_cutpoint %>% as.numeric

beta.post %>% melt(id.vars='samp_id') %>% 
  #filter(variable != 'pbto2_lo') %>% 
  dplyr::group_by(variable) %>% 
  summarise(
    lo=quantile(value, .025), 
    mid=quantile(value, .5), 
    hi=quantile(value, .975)
  ) %>% dplyr::mutate(variable=factor(variable)) %>% 
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed')

beta.post %>% ggplot(aes(x=pbto2_hi)) + geom_density()
beta.post[,c('pbto2_cp', 'pbto2_hi')] %>% ggplot(aes(x=pbto2_cp, y=pbto2_hi)) + geom_point()

