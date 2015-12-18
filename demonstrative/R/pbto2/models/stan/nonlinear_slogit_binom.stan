
data {
  int<lower=1> N_OBS;              // Total # of observations in time (~5k)
  int<lower=1> N_VARS;             // # of static covariates (~4)
  int<lower=2> N_UID;              // # patients (~300)
    
  int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=0, upper=1> y[N_UID];  // Vector of 0/1 outcomes
  matrix[N_UID, N_VARS] x;         // Static covariate matrix
  real z[N_OBS];                   // Time varying covariate
  real min_z;                      // Minimum value for lower center
  real max_z;                      // Maximum value for upper center
}

parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
  real<lower=-25, upper=0> betaz;
  real<lower=-25, upper=25> b;
  real<lower=min_z, upper=max_z> c;
}
transformed parameters {
  vector[N_UID] w;
  vector[N_UID] ct;
  real a1;
  real a2;

  ct <- rep_vector(0, N_UID);
  w <- rep_vector(0, N_UID);
  for (i in 1:N_OBS){
    w[uid[i]] <- w[uid[i]] + betaz / (1 + exp(b * (z[i] - c)));
    ct[uid[i]] <- ct[uid[i]] + 1;
  }
  for (i in 1:N_UID){
    w[i] <- w[i] / ct[i];
  }
}
model {
  real d;
  d <- 10;

  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sinh(b/d) ~ normal(0, 2);
  c ~ normal(0, 5);
  betaz ~ normal(0, 5); // Less regularized
  
  y ~ bernoulli_logit(alpha + x * beta + w);

  increment_log_prob(log(fabs((1/d)*cosh(b/d))));
}
