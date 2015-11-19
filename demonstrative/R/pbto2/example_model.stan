data {
  int<lower=1> N_OBS;                      // # of observations across all patients
  int<lower=1> N_VARS;                     // # of covariates excluding the one with a change point
  int<lower=2> N_UID;                      // # of patient IDs
  int<lower=1> N_CP;                       // # change points to marginalize over for Z

  int<lower=1>        uid[N_OBS];          // Vector of patient IDs for each observation
  int<lower=0, upper=1> y[N_UID];          // Vector of binary outcomes
  row_vector[N_VARS]    x[N_UID];          // Matrix of covariates NOT including covariate with change point (ie z)
  real                  z[N_OBS];          // Values for Z
  real        changepoints[N_CP];          // Change point values to marginalize over for Z
}
transformed data {
  real log_unif;
  log_unif <- -log(N_CP);
}
parameters {
  real alpha;
  vector[N_VARS] beta;   
  real beta_z_lo;
  real beta_z_hi;
}
transformed parameters {
  vector[N_CP] lp;
  lp <- rep_vector(log_unif, N_CP);

  // Compute log probability over all possible Z change points
  for (s in 1:N_CP){

    // These vectors will ultimately contain the percentage of observations where
    // Z is above and below the changepoint implied by s
    vector[N_UID] z_above;
    vector[N_UID] z_below;
    z_above <- rep_vector(0, N_UID);
    z_below <- rep_vector(0, N_UID);

    // First, tally the number of observations above and below changepoints[s]
    for (i in 1:N_OBS){
      if (z[i] > changepoints[s]){
        z_above[uid[i]] <- z_above[uid[i]] + 1;
      }else{
        z_below[uid[i]] <- z_below[uid[i]] + 1;
      }
    }

    // Finally, convert the counts of observations above and below cutpoints to percentages
    // and use those percentages as covariates to determine the log probability at this step for s 
    for (i in 1:N_UID){

      // Divide counts above and below by total count overall to give percentage
      z_above[i] <- z_above[i] / (z_above[i] + z_below[i]);
      z_below[i] <- z_below[i] / (z_above[i] + z_below[i]);

      
      lp[s] <- lp[s] + bernoulli_log(y[i], inv_logit(alpha + x[i] * beta + beta_z_lo * z_below[i] + beta_z_hi * z_above[i]));
    }
  }
}
model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 10);
  beta_z_lo ~ normal(0, 10);
  beta_z_hi ~ normal(0, 10);
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  int<lower=1, upper=N_CP> z_cp_idx;
  real z_cp;
  z_cp_idx <- categorical_rng(softmax(lp));
  z_cp <- changepoints[z_cp_idx];
}
