
data {
  int<lower=2> N_OUTCOME;          // # of possible GOS outcomes
  int<lower=1> N_OBS;              // Total # of observations (~10k)
  int<lower=1> N_VARS;             // # of static covariates (~4)
  int<lower=2> N_UID;              // # patients (~300)
    
  int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=1, upper=N_OUTCOME> y[N_UID];  // Vector of 0/1 outcomes
  matrix[N_UID, N_VARS] x;
  real z[N_OBS];                   // Time varying covariate
  real min_z;
  real max_z;
}

parameters {
  vector[N_VARS] beta; // Coefficients of static covariates
  ordered[N_OUTCOME - 1] outcome_cutpoints; 
  real<lower=0, upper=1> p;
  real<lower=-25, upper=0> betaz;
  real<lower=-25, upper=25> b2;
  real<lower=-25, upper=25> b1;
  real<lower=min_z, upper=0> c1;
  real<lower=0, upper=max_z> c2;
}
transformed parameters {
  vector[N_UID] w;
  vector[N_UID] ct;
  real a1;
  real a2;
  real c[2];

  c[1] <- c1;
  c[2] <- c2;

  a1 <- p * betaz;
  a2 <- (1 - p) * betaz;

  ct <- rep_vector(0, N_UID);
  w <- rep_vector(0, N_UID);
  for (i in 1:N_OBS){
    w[uid[i]] <- w[uid[i]] + a1 / (1 + exp(b1 * (z[i] - c[1]))) + a2 / (1 + exp(b2 * (z[i] - c[2])));
    ct[uid[i]] <- ct[uid[i]] + 1;
  }
  for (i in 1:N_UID){
    w[i] <- w[i] / ct[i];
  }
}
model {
  real y_hat[N_UID];
  real d;
  d <- 10;
  
  beta ~ normal(0, 5);
  sinh(b1/d) ~ normal(0, 2);
  sinh(b2/d) ~ normal(0, 2);
  c1 ~ normal(-1, 3);
  c2 ~ normal(1, 3);

  betaz ~ normal(0, 5); // Less regularized
  
  for (i in 1:N_UID)
    y[i] ~ ordered_logistic(x[i] * beta + w[i], outcome_cutpoints);

  increment_log_prob(log(fabs((1/d)*cosh(b1/d))));
  increment_log_prob(log(fabs((1/d)*cosh(b2/d))));
}

