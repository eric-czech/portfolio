data {
  int<lower=2> N_OUTCOME;                  // # of possible GOS outcomes
  int<lower=1> N_OBS;                      // # of observations
  int<lower=1> N_VARS;                     // # of covariates
  int<lower=1> N_CP;                       // # pbto2 cutpoints
  int<lower=1, upper=N_OUTCOME> y[N_OBS];  // Vector of outcomes
  row_vector[N_VARS] x[N_OBS];             // Matrix of covariates (eg age, marshall, gcs, pbto2)
}
parameters {
  vector[N_VARS] beta;
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
}
transformed parameters {
  vector[N_OBS] log_lik;
  for (i in 1:N_OBS){
    log_lik[i] <- ordered_logistic_log(y[i], x[i] * beta, outcome_cutpoints);
  }
}
model {
  beta ~ normal(0, 5);
  increment_log_prob(sum(log_lik));
}