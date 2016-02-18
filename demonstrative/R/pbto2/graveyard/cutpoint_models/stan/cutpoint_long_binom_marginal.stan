data {
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  int<lower=1> N_CP;                       // # pbto2 cutpoints
  int<lower=0, upper=1> y[N_OBS];          // Vector of 0/1 outcomes
  real pbto2[N_OBS];                       // Pbto2 values
  row_vector[N_VARS] x[N_OBS];             // Matrix of covariates w/o pbto2 (eg age, marshall, gcs)
  real pbto2_cutpoints[N_CP];
}
transformed data {
  real log_unif;
  log_unif <- -log(N_CP);
}
parameters {
  real alpha;
  vector[N_VARS] beta;
  real beta_pbto2_lo;
  real beta_pbto2_hi;
}
transformed parameters {
  vector[N_CP] lp;
  lp <- rep_vector(log_unif, N_CP);
  for (s in 1:N_CP){
    for (i in 1:N_OBS){
      if (pbto2[i] > pbto2_cutpoints[s]){
        lp[s] <- lp[s] + bernoulli_logit_log(y[i], alpha + x[i] * beta + beta_pbto2_hi * pbto2[i]);
      } else {
        lp[s] <- lp[s] + bernoulli_logit_log(y[i], alpha + x[i] * beta + beta_pbto2_lo * pbto2[i]);
      }
    }
  }
}
model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 10);
  beta_pbto2_lo ~ normal(0, 10);
  beta_pbto2_hi ~ normal(0, 10);
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  int<lower=1, upper=N_CP> pbto2_cutpoint_idx;
  real pbto2_cutpoint;
  
  pbto2_cutpoint_idx <- categorical_rng(softmax(lp));
  pbto2_cutpoint <- pbto2_cutpoints[pbto2_cutpoint_idx];
}
