data {
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  
  int<lower=0, upper=1> y[N_OBS];          // Vector of 0/1 outcomes
  real pbto2[N_OBS];                       // Pbto2 values
  row_vector[N_VARS] x[N_OBS];             // Matrix of covariates w/o pbto2 (eg age, marshall, gcs)
}
parameters {
  real alpha;
  vector[N_VARS] beta;
  real beta_pbto2_lo;
  real beta_pbto2_hi;
  real<lower=-10,upper=10> pbto2_cutpoint;
}
model {
  real y_hat[N_OBS];
  alpha ~ normal(0, 100);
  beta ~ normal(0, 10);
  beta_pbto2_lo ~ normal(0, 10);
  beta_pbto2_hi ~ normal(0, 10);
  
  for (i in 1:N_OBS)
    y_hat[i] <- inv_logit(alpha + x[i] * beta + if_else(pbto2[i] > pbto2_cutpoint, beta_pbto2_hi, beta_pbto2_lo) * pbto2[i]));
  y ~ bernoulli(y, y_hat);
}