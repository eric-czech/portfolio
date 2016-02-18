data {
  int<lower=2> N_OUTCOME;                  // # of possible GOS outcomes
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  int<lower=1, upper=N_OUTCOME> y[N_OBS];  // Vector of outcomes
  real pbto2[N_OBS];                       // Pbto2 values
  row_vector[N_VARS] x[N_OBS];             // Matrix of covariates w/o pbto2 (eg age, marshall, gcs)
}
parameters {
  vector[N_VARS] beta;
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
  real beta_pbto2_lo;
  real beta_pbto2_hi;
  real<lower=-10,upper=10> pbto2_cutpoint;
}
model {
  beta ~ normal(0, 10);
  beta_pbto2_lo ~ normal(0, 10);
  beta_pbto2_hi ~ normal(0, 10);
  
  for (i in 1:N_OBS){
    if (pbto2[i] < pbto2_cutpoint)
      y[i] ~ ordered_logistic(x[i] * beta + beta_pbto2_lo * pbto2[i], outcome_cutpoints);
    else
      y[i] ~ ordered_logistic(x[i] * beta + beta_pbto2_hi * pbto2[i], outcome_cutpoints);
  }
}
