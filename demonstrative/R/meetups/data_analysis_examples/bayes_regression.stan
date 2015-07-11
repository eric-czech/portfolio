# These are the quantities provided to the model:
data {
  int<lower=0> N; # Number of Homicide Rate observations over all countries
  int<lower=0> C; # Number of Countries
  int<lower=0> country[N];      # Country corresponding to each observation
  int<lower=0> year[N];         # Year for each data point
  real<lower=0.0> homicide[N];  # Homicide Rate for each data point
}

# These are the various parameters we'd like to estimate for the sake predicting things:
parameters {
  real beta0[C]; # Per-country intercept
  real beta1[C]; # Per-country slope
  real mu_beta0; # Average country intercept
  real mu_beta1; # Average country slope
  real<lower=0.00001> sigma_beta0;    # Per-country intercept standard deviation
  real<lower=0.00001> sigma_beta1;    # Per-country slope standard deviation
  real<lower=0.00001> sigma_homicide; # Noise variance (i.e. "epsilon")
}

# This is the "model" that relates data and parameters to one another:
model {

  # Country intercepts and slopes come from some diffuse normal distribution
  mu_beta0 ~ normal(0, 100); 
  mu_beta1 ~ normal(0, 100);
  
  # Per-country slopes and intercepts share above means
  beta0 ~ normal(mu_beta0, sigma_beta0);
  beta1 ~ normal(mu_beta1, sigma_beta1); 
  
  for (i in 1:N){
    homicide[i] ~ normal(beta0[country[i]] + beta1[country[i]] * year[i], sigma_homicide);
  }
}

# These are the predictions we'd like to take away from the model:
generated quantities {
  vector[C] predictions;
  for (i in 1:C){
    predictions[i] <- normal_rng(beta0[i] + beta1[i] * 15, sigma_homicide);
  }
}