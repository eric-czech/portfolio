// This logistic regression model contains a binary outcome specific to about 300 separate patients
// as well as for each of those patients, a set of static covariates that do not vary in time (called "x")
// and a single time-varying covariate called "z".  The overall number of observations is around 10k but in reality
// that means there are ~10k time-varying measurements across the 300 patients as well as 300 values for the other
// static covariates.  The model is then trying to determine a change point in the "z" variable where the number of 
// measurements above and below that change point are allowed to have different effects on the outcome.

data {
  int<lower=1> N_OBS;              // Total # of observations (~10k)
  int<lower=1> N_VARS;             // # of static covariates (~4)
  int<lower=2> N_UID;              // # patients (~300)
  int<lower=1> N_CP;               // # cutpoints for z (~100)

  int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=0, upper=1> y[N_UID];  // Vector of 0/1 outcomes
  row_vector[N_VARS] x[N_UID];     // Matrix of static covariates
  real z[N_OBS];                   // Time varying covariate
  real z_cutpoints[N_CP];          // Change point values for Z to marginalize over
}
transformed data {
  real log_unif;
  matrix[N_CP, N_UID] z_above;
  matrix[N_CP, N_UID] z_below;

  log_unif <- -log(N_CP);

  z_above <- rep_matrix(0, N_CP, N_UID);
  z_below <- rep_matrix(0, N_CP, N_UID);  
  for (s in 1:N_CP){
    // Loop over all ~10k observations and tally the number of z values above
    // and below the change point for each patient
    for (i in 1:N_OBS){
      if (z[i] > z_cutpoints[s]){
        z_above[s, uid[i]] <- z_above[s, uid[i]] + 1;
      }else{
        z_below[s, uid[i]] <- z_below[s, uid[i]] + 1;
      }
    }
  }
}
parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
  real beta_z_lo;      // Coefficient for number of z measurements below change point
  real beta_z_hi;      // Coefficient for number of z measurements above change point
}
transformed parameters {
  vector[N_CP] lp;

  // Give change points uniform prior
  lp <- rep_vector(log_unif, N_CP);

  // Loop over change points and accumulate log probability of each
  for (s in 1:N_CP){
    vector[N_UID] y_hat;

    // Use the tallys calculated above to determine the contribution to 
    // log probability for each patient
    for (i in 1:N_UID){
      y_hat[i] <- alpha + x[i] * beta + beta_z_lo * z_below[s, i] + beta_z_hi * z_above[s, i];
    }
    lp[s] <- lp[s] + bernoulli_logit_log(y, y_hat);
  }
}
model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 5);
  beta_z_lo ~ normal(0, 5);
  beta_z_hi ~ normal(0, 5);
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  // Sample the cutpoint value for z on this step
  int<lower=1, upper=N_CP> z_cutpoint_idx;
  real z_cutpoint;
  z_cutpoint_idx <- categorical_rng(softmax(lp));
  z_cutpoint <- z_cutpoints[z_cutpoint_idx];
}