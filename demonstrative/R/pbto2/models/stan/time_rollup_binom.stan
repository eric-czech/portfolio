data {
  int<lower=1> N_OBS;              // Total # of observations (~10k)
  int<lower=1> N_VARS;             // # of static covariates (~4)
  int<lower=2> N_UID;              // # patients (~300)
  int<lower=1> N_Z_CP;             // # cutpoints for z (~100)
  int<lower=1> N_T_CP;             // # cutpoints for t (~10)
    
  int<lower=1> uid[N_OBS];         // Patient id vector
  int<lower=0, upper=1> y[N_UID];  // Vector of 0/1 outcomes
  row_vector[N_VARS] x[N_UID];     // Matrix of static covariates
  real z[N_OBS];                   // Time varying covariate
  real z_cutpoints[N_Z_CP];          // Change point values for Z to marginalize over
  int<lower=0> t[N_OBS];
  int<lower=0> t_cutpoints[N_T_CP]; 
}
transformed data {
  real log_unif;
  int<lower=0> z_above[N_Z_CP, N_T_CP, N_UID];
  int<lower=0> z_below[N_Z_CP, N_T_CP, N_UID];
  
  log_unif <- -log(N_Z_CP * N_T_CP);
  for (zi in 1:N_Z_CP){
    for (ti in 1:N_T_CP){
      for (ui in 1:N_UID){
        z_above[zi, ti, ui] <- 0;
        z_below[zi, ti, ui] <- 0;
      }
    }
  }
    
  for (zi in 1:N_Z_CP){
    for (ti in 1:N_T_CP){
      // Loop over all ~10k observations and tally the number of z values above
      // and below the change point for each patient
      for (i in 1:N_OBS){
        if (t[i] <= t_cutpoints[ti]){
          if (z[i] > z_cutpoints[zi]){
            z_above[zi, ti, uid[i]] <- z_above[zi, ti, uid[i]] + 1;
          }else{
            z_below[zi, ti, uid[i]] <- z_below[zi, ti, uid[i]] + 1;
          }
        }
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
  vector[N_Z_CP] zlp;
  vector[N_T_CP] tlp;
  vector[] lp;
  
  // Give change points uniform prior
  lp <- rep_vector(-log(N_Z_CP * N_T_CP), N_Z_CP * N_T_CP);
  tlp <- rep_vector(-log(N_T_CP), N_T_CP);
  
  // Loop over change points and accumulate log probability of each
  for (zi in 1:N_Z_CP){
    for (ti in 1:N_T_CP){
      vector[N_UID] y_hat;
      real lp;
      
      // Use the tallys calculated above to determine the contribution to 
      // log probability for each patient
      for (i in 1:N_UID){
        y_hat[i] <- alpha + x[i] * beta + beta_z_lo * z_below[zi, ti, i] + beta_z_hi * z_above[zi, ti, i];
      }
      lp <- bernoulli_logit_log(y, y_hat);
      zlp[zi] <- zlp[zi] + lp;
      tlp[ti] <- tlp[ti] + lp;
      p <- p + exp(lp);
    }
  }
}
model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 5);
  beta_z_lo ~ normal(0, 5);
  beta_z_hi ~ normal(0, 5);
  increment_log_prob(log(p));
}
generated quantities {
  // Sample the cutpoint values for z and t on this step
  int<lower=1, upper=N_Z_CP> z_cutpoint_idx;
  int<lower=1, upper=N_T_CP> t_cutpoint_idx;
  real z_cutpoint;
  int t_cutpoint;
  
  z_cutpoint_idx <- categorical_rng(softmax(zlp));
  z_cutpoint <- z_cutpoints[z_cutpoint_idx];
  t_cutpoint_idx <- categorical_rng(softmax(tlp));
  t_cutpoint <- t_cutpoints[t_cutpoint_idx];
}
