library(foreach)
library(dplyr)
library(ggplot2)
library(MASS)
library(rstan)
library(reshape2)

#d <- read.csv('/Users/eczech/data/ptbo2/export/data_modeling_no_gos_interp.csv')
#d <- read.csv('/Users/eczech/data/ptbo2/export/data_modeling_w_gos_interp.csv')
d <- read.csv('/Users/eczech/data/ptbo2/export/data_modeling.csv')


features <- d %>% dplyr::select(-outcome) %>% names
primary.feat <- 'pbto2_p5'
secondary.feat <- features[features != primary.feat]
  
d %>% ggplot(aes(x=factor(outcome), y=pbto2_p10)) + geom_boxplot()

### Polr

d.polr <- d %>% mutate(outcome=factor(outcome, levels=1:5, labels=c('Dead', 'Bad1', 'Bad2', 'Good1', 'Good2'))) 
m.polr <- polr(outcome ~ pbto2_mean_under_15  + age + marshall + gcs + sex, data = d.polr, Hess=TRUE)
coefs <- coef(summary(m.polr))
p <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
coefs <- cbind(coefs, "p value" = p)
coefs

primary.feat.v <- seq(min(d[,primary.feat]), max(d[,primary.feat]), length.out=30)
d.pred <- foreach(v=primary.feat.v, .combine=rbind) %do% {
  r <- d[1,secondary.feat] * 0
  r[,primary.feat] <- v
  r
}
d.pred <- predict(m.polr, d.pred, type='p') %>% as.data.frame
d.pred[,'pbto2'] = primary.feat.v
d.pred %>% melt(id.vars='pbto2') %>% 
  ggplot(aes(x=pbto2, y=value, fill=variable)) + geom_bar(stat='identity', position='stack')

### GLM 

d.glm <- d %>% mutate(outcome=factor(ifelse(outcome <= 3, 'Bad', 'Good'), levels=c('Bad', 'Good')))
m.glm <- glm(outcome ~ pbto2_p5 + age + marshall + gcs + sex, data = d.glm, family='binomial')

### Stan

stan.model <- '
  data {
    int<lower=2> N_OUTCOME;                  // # of possible GOS outcomes 
    int<lower=0> N_OBS;                      // # of observations
    int<lower=1> N_VARS;                     // # of covariates
    int<lower=1, upper=N_OUTCOME> y[N_OBS];  // Vector of outcomes
    row_vector[N_VARS] x[N_OBS];             // Matrix of covariates
  }
  parameters {
    vector[N_VARS] beta;   // Slope estimates 
    ordered[N_OUTCOME - 1] cutpoints; 
  }
  model {
    // Assign normal priors to coefficients to estimate
    beta ~ normal(0, 10);
    
    // Create model where the output is predicted by the per-job-title 
    // intercept plus the per-job-title slope times each personality trait
    for (i in 1:N_OBS)
      y[i] ~ ordered_logistic(x[i] * beta, cutpoints);
  }
'

d.stan <- d %>% dplyr::select_(.dots=c(features, 'outcome')) 

# Convert the raw data created into Stan data
d.model <- list(
  N_OUTCOME = length(unique(d.stan$outcome)), 
  N_OBS = nrow(d.stan),
  N_VARS = length(features),
  y = as.integer(d.stan$outcome),
  x = d.stan %>% dplyr::select(-outcome)
)

# Run the sampler
fit <- stan(model_code = stan.model, data = d.model,
            warmup = 100, iter = 1000, thin = 5, chains = 2, verbose = FALSE)

rstan::traceplot(fit)
plot(fit)


post <- rstan::extract(fit)
beta.post <- data.frame(post$beta) %>% setNames(features) %>% dplyr::mutate(samp_id=1:nrow(.))

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

### Coefficients from all models

stan.coef <- beta.post %>% melt(id.vars='samp_id') %>% 
  dplyr::group_by(variable) %>% 
  summarise(
    lo=quantile(value, .025), 
    mid=quantile(value, .5), 
    hi=quantile(value, .975)
  )
get.glm.coefs <- function(m){
  cbind(coef(m), sqrt(diag(vcov(m))))[-1,] %>% data.frame %>% setNames(c('mid', 'se'))
}
get.polr.coefs <- function(m){
  coefs <- coef(m)
  vcoef <- sqrt(diag(vcov(m)))[1:length(coefs)]
  cbind(coefs, vcoef) %>% data.frame %>% setNames(c('mid', 'se'))
}
get.coef.range <- function(coefs, q=1.96){
  coefs[,'lo'] <- coefs[,'mid'] - q * coefs[,'se']
  coefs[,'hi'] <- coefs[,'mid'] + q * coefs[,'se']
  coefs %>% dplyr::select(-se) %>% add_rownames(var = 'variable')
}
#get.coef.range(get.glm.coefs(m.glm))
#get.coef.range(get.polr.coefs(m.polr))
all.coef <- rbind(
  get.coef.range(get.polr.coefs(m.polr)) %>% mutate(model='POLR'),
  get.coef.range(get.glm.coefs(m.glm)) %>% mutate(model='GLM'),
  stan.coef %>% mutate(model='STAN')
)
coef.levels <- all.coef %>% group_by(variable) %>% 
  summarise(mean=mean(mid)) %>% ungroup %>% arrange(mean) %>% .$variable
all.coef %>% mutate(variable = factor(variable, levels=coef.levels)) %>%
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=model)) + 
  geom_pointrange(position = position_dodge(width = 0.4)) + theme_bw() +
  ggtitle('LogReg Coefficient Estimates\n(With GOS Interpolation)') +
  xlab('Variable') + ylab('Coefficient 95% Credible/Confidence Interval') + 
  ggsave('/Users/eczech/repos/portfolio/demonstrative/R/pbto2/images/coefs_w_interp.png')



# s1 <- stan_model(model_code = code) # compile the model
# 
# m1 <- sampling(object = s1, data = df1, chains = 1,
#                seed = seed, chain_id = 1, iter = 1000) 
# m2 <- sampling(object = s1, data = df2, chains = 1,
#                seed = seed, chain_id = 2, iter = 1000)
# 
# f12 <- sflist2stanfit(list(m1, m2))