library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

#rstan_options(auto_write = TRUE)
#options(mc.cores = parallel::detectCores())

d <- read.csv('/Users/eczech/data/ptbo2/export/data_long_cutpoint.csv', stringsAsFactors=F)

#features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')
features <- c('pbto2', 'sex', 'marshall')
scale <- function(x) (x - mean(x)) / sd(x)
d.stan <- d %>% 
  mutate(outcome=gos.3.binary) %>% 
  mutate_each_(funs(scale), features[features != 'pbto2']) %>%
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
  int<lower=0, upper=1> y[N_UID];  // Vector of outcomes
  row_vector[N_VARS] x[N_UID];           // Matrix of covariates (eg age, marshall, gcs)
  real pbto2[N_OBS]; 
  int<lower=1> N_CP; // # pbto2 cutpoints
  real pbto2_cutpoints[N_CP];

}
transformed data {
  real log_unif;
  log_unif <- -log(N_CP);
}
parameters {
  vector[N_VARS] beta;   
  real beta_pbto2_lo;
  real beta_pbto2_hi;
}
transformed parameters {
  vector[N_CP] lp;
  lp <- rep_vector(log_unif, N_CP);
  for (s in 1:N_CP){
    vector[N_UID] x_above;
    vector[N_UID] x_below;
    vector[N_UID] n_above;
    vector[N_UID] n_below;
    x_above <- rep_vector(0, N_UID);
    x_below <- rep_vector(0, N_UID);
    n_above <- rep_vector(0, N_UID);
    n_below <- rep_vector(0, N_UID);
    for (i in 1:N_OBS){
      if (pbto2[i] > pbto2_cutpoints[s]){
        n_above[uid[i]] <- n_above[uid[i]] + 1;
        x_above[uid[i]] <- x_above[uid[i]] + pbto2[i];
      }else{
        n_below[uid[i]] <- n_below[uid[i]] + 1;
        x_below[uid[i]] <- x_below[uid[i]] + pbto2[i];
      }
    }
    for (i in 1:N_UID){
      if (n_above[i] == 0)
        x_above[i] <- pbto2_cutpoints[s];
      else
        x_above[i] <- x_above[i] / n_above[i];
      if (n_below[i] == 0)
        x_below[i] <- pbto2_cutpoints[s];
      else
        x_below[i] <- x_below[i] / n_below[i];
    }
    for (i in 1:N_UID){
      lp[s] <- lp[s] + bernoulli_log(y[i], inv_logit(x[i] * beta + beta_pbto2_lo * x_below[i] + beta_pbto2_hi * x_above[i]));
    }
  }
}
model {
  beta ~ normal(0, 3);
  beta_pbto2_lo ~ normal(0, 3);
  beta_pbto2_hi ~ normal(0, 3);
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  int<lower=1, upper=N_CP> s;
  s <- categorical_rng(softmax(lp));
}
'

d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid)
n.cp <- 50
# Convert the raw data created into Stan data
d.model <- list(
  N_OUTCOME = length(unique(d.stan$outcome)), 
  N_OBS = nrow(d.stan),
  N_VARS = length(features) - 1,
  N_UID = max(d.stan$uid),
  N_CP = n.cp,
  y = d.stan.uid %>% .$outcome %>% as.integer,
  x = d.stan.uid %>% dplyr::select(-outcome, -pbto2, -uid),
  pbto2 = d.stan$pbto2,
  uid = d.stan$uid,
  pbto2_cutpoints = seq(1, 30, length.out=n.cp)
)

# Run the sampler
init_fun <- function() { list(
  beta=rep(0, d.model$N_VARS),
  beta_pbto2_lo=0,
  beta_pbto2_hi=0,
  outcome_cutpoints=c(0)
)} 
fit <- stan(model_code = stan.model, data = d.model,
            warmup = 100, iter = 3000, thin = 10, chains = 1, verbose = FALSE, init=init_fun)

rstan::traceplot(fit)
plot(fit)


post <- rstan::extract(fit)
beta.post <- data.frame(post$beta) %>% setNames(features[-1]) %>% dplyr::mutate(samp_id=1:nrow(.))
beta.post[,'pbto2_lo'] <- post$beta_pbto2_lo %>% as.numeric
beta.post[,'pbto2_hi'] <- post$beta_pbto2_hi %>% as.numeric
beta.post[,'pbto2_cp'] <- sapply(post$s %>% as.numeric, function(i) d.model$pbto2_cutpoints[i])


dev.off(); hist(beta.post$pbto2_cp)
beta.post %>% melt(id.vars='samp_id') %>% 
  filter(variable != 'pbto2_cp') %>% 
  dplyr::group_by(variable) %>% 
  summarise(
    lo=quantile(value, .025), 
    mid=quantile(value, .5), 
    hi=quantile(value, .975)
  ) %>% dplyr::mutate(variable=factor(variable)) %>% 
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed')

