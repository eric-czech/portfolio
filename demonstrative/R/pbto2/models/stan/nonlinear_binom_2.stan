
data {
  int<lower=1> N_OBS;              // Total # of observations (~10k)
  int<lower=1> N_VARS;             // # of static covariates (~4)
    int<lower=2> N_UID;              // # patients (~300)
    
    int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=0, upper=1> y[N_UID];  // Vector of 0/1 outcomes
  //row_vector[N_VARS] x[N_UID];     // Matrix of static covariates
  matrix[N_UID, N_VARS] x;
  real z[N_OBS];                   // Time varying covariate
}

parameters {
  real alpha;          // Intercept for logit model
  vector[N_VARS] beta; // Coefficients of static covariates
  real<lower=0, upper=1> p;
  real<lower=-100, upper=100> betaz;
  real<lower=-25, upper=25> b2;
  real<lower=-25, upper=25> b1;
  real<lower=-25, upper=0> c1;
  real<lower=0, upper=25> c2;
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
  alpha ~ normal(0, 3);
  beta ~ normal(0, 3);
  //b1 ~ double_exponential(0, 10);
  //b2 ~ double_exponential(0, 10);
  c1 ~ normal(-2, 1);
  c2 ~ normal(2, 1);
  betaz ~ normal(0, 3);
  
  y ~ bernoulli_logit(alpha + x * beta + w);
}
