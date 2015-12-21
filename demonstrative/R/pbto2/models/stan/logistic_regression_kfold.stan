data {
  int<lower=2> N_OBS;              // Total # of observations in time 
  int<lower=2> N_OBS_HO;           // Total # of observations in time (hold-out)
  
  int<lower=1> N_VARS;             // # of static covariates (~4)
  
  int<lower=0, upper=1> y[N_OBS];       // Vector of 0/1 outcomes
  int<lower=0, upper=1> y_ho[N_OBS_HO]; // Vector of 0/1 outcomes (hold-out)
  
  matrix[N_OBS, N_VARS] x;         // Static covariate matrix
  matrix[N_OBS_HO, N_VARS] x_ho;   // Static covariate matrix (hold-out)
}
parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + x * beta);
}
generated quantities {
  vector[N_OBS] log_lik;
  vector[N_OBS_HO] y_pred;
  for (i in 1:N_OBS_HO)
    y_pred[i] <- inv_logit(alpha + x_ho[i] * beta);
  for (i in 1:N_OBS)
    log_lik[i] <- bernoulli_logit_log(y[i], alpha + x[i] * beta);
}