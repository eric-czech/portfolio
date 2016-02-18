library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

d <- read.csv('/Users/eczech/data/ptbo2/export/data_long_cutpoint_48hr.csv', stringsAsFactors=F)

features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')
#features <- c('pbto2', 'sex', 'marshall')
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
  int<lower=1, upper=N_OUTCOME> y[N_UID];  // Vector of outcomes
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
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
  real beta_pbto2_lo;
  real beta_pbto2_hi;
}
transformed parameters {
  vector[N_CP] lp;
  lp <- rep_vector(log_unif, N_CP);
  for (s in 1:N_CP){
    vector[N_UID] x_above;
    vector[N_UID] x_below;
    x_above <- rep_vector(0, N_UID);
    x_below <- rep_vector(0, N_UID);
    for (i in 1:N_OBS){
      if (pbto2[i] > pbto2_cutpoints[s]){
        x_above[uid[i]] <- x_above[uid[i]] + 1;
      }else{
        x_below[uid[i]] <- x_below[uid[i]] + 1;
      }
    }
    for (i in 1:N_UID){
      //x_above[i] <- x_above[i] / (x_above[i] + x_below[i]);
      //x_below[i] <- x_below[i] / (x_above[i] + x_below[i]);
      lp[s] <- lp[s] + ordered_logistic_log(y[i], x[i] * beta + beta_pbto2_lo * x_below[i] + beta_pbto2_hi * x_above[i], outcome_cutpoints);
    }
  }
}
model {
  beta ~ normal(0, 5);
  beta_pbto2_lo ~ normal(0, 5);
  beta_pbto2_hi ~ normal(0, 5);
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  int<lower=1, upper=N_CP> pbto2_cutpoint_idx;
  real pbto2_cutpoint;
  pbto2_cutpoint_idx <- categorical_rng(softmax(lp));
  pbto2_cutpoint <- pbto2_cutpoints[pbto2_cutpoint_idx];
}
'

d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid)
n.cp <- 150
scale.cutpoint <- function(x) (x - mean(d$pbto2)) / sd(d$pbto2)

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
  pbto2_cutpoints = scale.cutpoint(seq(1, 150, length.out=n.cp))
)

# Run the sampler
init_fun <- function() { list(
  beta=rep(0, d.model$N_VARS),
  beta_pbto2_lo=0,
  beta_pbto2_hi=0,
  outcome_cutpoints=c(-.5, .5)
)} 
posterior <- stan(model_code = stan.model, data = d.model,
            warmup = 100, iter = 5000, thin = 3, chains = 1, verbose = FALSE, init=init_fun)

rstan::traceplot(posterior, c('beta', 'beta_pbto2_lo', 'beta_pbto2_hi', 'pbto2_cutpoint_idx', 'pbto2_cutpoint'))
plot(posterior)


post <- rstan::extract(posterior)
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


