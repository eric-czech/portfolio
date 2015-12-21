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

  real z1[N_OBS];
  real z_ho1[N_OBS_HO];
  real z2[N_OBS];
  real z_ho2[N_OBS_HO];

  real min_z1;
  real max_z1;
  real min_z2;
  real max_z2;

  real NA_VALUE;  // Sentinel value
}

parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
  real<lower=0, upper=1> p1;
  real<lower=0, upper=1> p2;
  real<lower=-25, upper=0> betaz1;
  real<lower=-25, upper=0> betaz2;
  real<lower=-25, upper=25> b11;
  real<lower=-25, upper=25> b12;
  real<lower=-25, upper=25> b21;
  real<lower=-25, upper=25> b22;
  real<lower=min_z1, upper=0> c11;
  real<lower=min_z2, upper=0> c12;
  real<lower=0, upper=max_z1> c21;
  real<lower=0, upper=max_z2> c22;
}
transformed parameters {
  vector[N_UID] w1;
  vector[N_UID_HO] w_ho1;
  vector[N_UID] ct1;
  vector[N_UID_HO] ct_ho1;
  vector[N_UID] w2;
  vector[N_UID_HO] w_ho2;
  vector[N_UID] ct2;
  vector[N_UID_HO] ct_ho2;
  real a11;
  real a21;
  real a12;
  real a22;

  a11 <- p1 * betaz1;
  a21 <- (1 - p1) * betaz1;
  a12 <- p2 * betaz2;
  a22 <- (1 - p2) * betaz2;

  ct1 <- rep_vector(0, N_UID);
  ct_ho1 <- rep_vector(0, N_UID_HO);
  w1 <- rep_vector(0, N_UID);
  w_ho1 <- rep_vector(0, N_UID_HO);
  ct2 <- rep_vector(0, N_UID);
  ct_ho2 <- rep_vector(0, N_UID_HO);
  w2 <- rep_vector(0, N_UID);
  w_ho2 <- rep_vector(0, N_UID_HO);

  for (i in 1:N_OBS){
    if (z1[i] != NA_VALUE){
      w1[uid[i]] <- w1[uid[i]] + a11 / (1 + exp(b11 * (z1[i] - c11))) + a21 / (1 + exp(b21 * (z1[i] - c21)));
      ct1[uid[i]] <- ct1[uid[i]] + 1;
    } 
    if (z2[i] != NA_VALUE){
      w2[uid[i]] <- w2[uid[i]] + a12 / (1 + exp(b12 * (z2[i] - c12))) + a22 / (1 + exp(b22 * (z2[i] - c22)));
      ct2[uid[i]] <- ct2[uid[i]] + 1;
    }
  }
  for (i in 1:N_UID){
    if (ct1[i] > 0)
      w1[i] <- w1[i] / ct1[i];
    if (ct2[i] > 0)
      w2[i] <- w2[i] / ct2[i];
  }
  for (i in 1:N_OBS_HO){
    if (z_ho1[i] != NA_VALUE){
      w_ho1[uid_ho[i]] <- w_ho1[uid_ho[i]] + a11 / (1 + exp(b11 * (z_ho1[i] - c11))) + a21 / (1 + exp(b21 * (z_ho1[i] - c21)));
      ct_ho1[uid_ho[i]] <- ct_ho1[uid_ho[i]] + 1;
    }
    if (z_ho2[i] != NA_VALUE){
      w_ho2[uid_ho[i]] <- w_ho2[uid_ho[i]] + a12 / (1 + exp(b12 * (z_ho2[i] - c12))) + a22 / (1 + exp(b22 * (z_ho2[i] - c22)));
      ct_ho2[uid_ho[i]] <- ct_ho2[uid_ho[i]] + 1;
    }
  }
  for (i in 1:N_UID_HO){
    if (ct_ho1[i] > 0)
      w_ho1[i] <- w_ho1[i] / ct_ho1[i];
    if (ct_ho2[i] > 0)
      w_ho2[i] <- w_ho2[i] / ct_ho2[i];
  }
}
model {
  real d;
  d <- 10;

  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sinh(b11/d) ~ normal(0, 2);
  sinh(b21/d) ~ normal(0, 2);
  sinh(b12/d) ~ normal(0, 2);
  sinh(b22/d) ~ normal(0, 2);
  c11 ~ normal(-1, 3);
  c21 ~ normal(1, 3);
  c12 ~ normal(-1, 3);
  c22 ~ normal(1, 3);
  betaz1 ~ normal(0, 5); 
  betaz2 ~ normal(0, 5); 
  
  y ~ bernoulli_logit(alpha + x * beta + w1 + w2);

  increment_log_prob(log(fabs((1/d)*cosh(b11/d))));
  increment_log_prob(log(fabs((1/d)*cosh(b21/d))));
  increment_log_prob(log(fabs((1/d)*cosh(b12/d))));
  increment_log_prob(log(fabs((1/d)*cosh(b22/d))));
}
generated quantities {
  vector[N_UID] log_lik;
  vector[N_UID_HO] y_pred;
  for (i in 1:N_UID_HO)
    y_pred[i] <- inv_logit(alpha + x_ho[i] * beta + w_ho1[i] + w_ho2[i]);
  for (i in 1:N_UID)
    log_lik[i] <- bernoulli_logit_log(y[i], alpha + x[i] * beta + w1[i] + w2[i]);
}