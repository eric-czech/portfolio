data {
  int<lower=2> N_OBS;              // Total # of observations in time 
  int<lower=2> N_OBS_HO;           // Total # of observations in time (hold-out)

  int<lower=1> N_VARS;             // # of static covariates (~4)

  int<lower=2> N_UID;              // # patients 
  int<lower=2> N_UID_HO;           // # patients (hold-out)
    
  int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=1> uid_ho[N_OBS_HO];      // Patient id vector (hold-out)

  int<lower=0, upper=1> y[N_UID];       // Vector of 0/1 outcomes
  int<lower=0, upper=1> y_ho[N_UID_HO]; // Vector of 0/1 outcomes (hold-out)

  matrix[N_UID, N_VARS] x;         // Static covariate matrix
  matrix[N_UID_HO, N_VARS] x_ho;   // Static covariate matrix (hold-out)

  real z[N_OBS];                   // Time varying covariate
  real z_ho[N_OBS_HO];             // Time varying covariate (hold-out)

  real min_z;                      // Minimum value for lower center
  real max_z;                      // Maximum value for upper center
}

parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
  real<lower=0, upper=1> p;
  real<lower=-25, upper=0> betaz;
  real<lower=-25, upper=25> b2;
  real<lower=-25, upper=25> b1;
  real<lower=min_z, upper=0> c1;
  real<lower=0, upper=max_z> c2;
}
transformed parameters {
  vector[N_UID] w;
  vector[N_UID_HO] w_ho;
  vector[N_UID] ct;
  vector[N_UID_HO] ct_ho;

  real a1;
  real a2;
  real c[2];

  c[1] <- c1;
  c[2] <- c2;

  a1 <- p * betaz;
  a2 <- (1 - p) * betaz;

  ct <- rep_vector(0, N_UID);
  ct_ho <- rep_vector(0, N_UID_HO);
  w <- rep_vector(0, N_UID);
  w_ho <- rep_vector(0, N_UID_HO);

  for (i in 1:N_OBS){
    w[uid[i]] <- w[uid[i]] + a1 / (1 + exp(b1 * (z[i] - c[1]))) + a2 / (1 + exp(b2 * (z[i] - c[2])));
    ct[uid[i]] <- ct[uid[i]] + 1;
  }
  for (i in 1:N_UID){
    w[i] <- w[i] / ct[i];
  }

  for (i in 1:N_OBS_HO){
    w_ho[uid_ho[i]] <- w_ho[uid_ho[i]] + a1 / (1 + exp(b1 * (z_ho[i] - c[1]))) + a2 / (1 + exp(b2 * (z_ho[i] - c[2])));
    ct_ho[uid_ho[i]] <- ct_ho[uid_ho[i]] + 1;
  }
  for (i in 1:N_UID_HO){
    w_ho[i] <- w_ho[i] / ct_ho[i];
  }
}
model {
  real d;
  d <- 10;

  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sinh(b1/d) ~ normal(0, 2);
  sinh(b2/d) ~ normal(0, 2);
  c1 ~ normal(-1, 3);
  c2 ~ normal(1, 3);
  betaz ~ normal(0, 5); 
  
  y ~ bernoulli_logit(alpha + x * beta + w);

  increment_log_prob(log(fabs((1/d)*cosh(b1/d))));
  increment_log_prob(log(fabs((1/d)*cosh(b2/d))));
}
generated quantities {
  vector[N_UID] log_lik;
  vector[N_UID_HO] y_pred;
  for (i in 1:N_UID_HO)
    y_pred[i] <- inv_logit(alpha + x_ho[i] * beta + w_ho[i]);
  for (i in 1:N_UID)
    log_lik[i] <- bernoulli_logit_log(y[i], alpha + x[i] * beta + w[i]);
}