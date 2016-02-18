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
  dplyr::select_(.dots=c(features, 'outcome')) 


### Stan

stan.model <- '
data {
  int<lower=2> N_OUTCOME;                  // # of possible GOS outcomes 
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  int<lower=1, upper=N_OUTCOME> y[N_OBS];  // Vector of outcomes
  real pbto2[N_OBS];      
  row_vector[N_VARS] x[N_OBS];           // Matrix of covariates (eg age, marshall, gcs)
}
parameters {
  vector[N_VARS] beta;   // Slope estimates 
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
  real beta_pbto2_lo;
  real beta_pbto2_hi;
  real<lower=-2,upper=2> pbto2_cutpoint;
}
model {

  // Assign normal priors to coefficients to estimate
  beta ~ normal(0, 10);
  beta_pbto2_lo ~ normal(0, 10);
  beta_pbto2_hi ~ normal(0, 10);
  pbto2_cutpoint ~ normal(0, 10);

  
  for (i in 1:N_OBS){
    if (pbto2[i] < pbto2_cutpoint)
      y[i] ~ ordered_logistic(x[i] * beta + beta_pbto2_lo * pbto2[i], outcome_cutpoints);
    else
      y[i] ~ ordered_logistic(x[i] * beta + beta_pbto2_hi * pbto2[i], outcome_cutpoints);
  }
}
'

# Convert the raw data created into Stan data
d.model <- list(
  N_OUTCOME = length(unique(d.stan$outcome)), 
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  y = as.integer(d.stan$outcome),
  x = d.stan %>% dplyr::select(-outcome, -pbto2),
  pbto2 = d.stan$pbto2
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
            warmup = 500, iter = 3000, thin = 5, chains = 2, verbose = FALSE)
# 17448.1 seconds (Total)

rstan::traceplot(fit)
plot(fit)


post <- rstan::extract(fit)
beta.post <- data.frame(post$beta) %>% setNames(features[-1]) %>% dplyr::mutate(samp_id=1:nrow(.))
beta.post$pbto2_cp <- post$pbto2_cutpoint %>% as.numeric
beta.post$pbto2_hi <- post$beta_pbto2_hi %>% as.numeric
beta.post$pbto2_lo <- post$beta_pbto2_lo %>% as.numeric

unscale <- function(x, var) x * sd(d[,var]) + mean(d[,var])

plot(density(unscale(beta.post$pbto2_cp, 'pbto2')))
hist(unscale(beta.post$pbto2_cp, 'pbto2'), breaks = 50)
beta.post %>% filter(pbto2_cp > -1) %>% ggplot(aes(x=pbto2_cp, y=pbto2_lo)) + geom_point()

beta.post %>% melt(id.vars='samp_id') %>% 
  dplyr::group_by(variable) %>% 
  summarise(
    lo=quantile(value, .025), 
    mid=quantile(value, .5), 
    hi=quantile(value, .975)
  ) %>% dplyr::mutate(variable=factor(variable)) %>% 
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed')



