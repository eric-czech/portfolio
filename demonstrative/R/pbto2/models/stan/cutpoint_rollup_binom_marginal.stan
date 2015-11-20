data {
  int<lower=1> N_OBS;              // # of observations
  int<lower=1> N_VARS;             // # of covariates
  int<lower=2> N_UID;              // # patients
  int<lower=1> uid[N_OBS];
  int<lower=0, upper=1> y[N_UID];  // Vector of outcomes
  row_vector[N_VARS] x[N_UID];     // Matrix of covariates (eg age, marshall, gcs)
  real pbto2[N_OBS]; 
  int<lower=1> N_CP;               // # pbto2 cutpoints
  real pbto2_cutpoints[N_CP];
}
transformed data {
  real log_unif;
  log_unif <- -log(N_CP);
}
parameters {
  vector[N_VARS] beta;
  real alpha;
  real beta_pbto2_lo;
  real beta_pbto2_hi;
}
transformed parameters {
  vector[N_CP] lp;
  lp <- rep_vector(log_unif, N_CP);
  for (s in 1:N_CP){
    vector[N_UID] x_above;
    vector[N_UID] x_below;
    vector[N_UID] y_hat;
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
      y_hat[i] <- alpha + x[i] * beta + beta_pbto2_lo * x_below[i] + beta_pbto2_hi * x_above[i];
    }
    lp[s] <- lp[s] + bernoulli_logit_log(y, y_hat);
  }
}
model {
  alpha ~ normal(0, 100);
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