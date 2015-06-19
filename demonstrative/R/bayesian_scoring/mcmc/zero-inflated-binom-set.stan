data {
  int<lower=0> N; // Number of days
  int<lower=0> G; // Number of groups
  int<lower=0> y[G, N]; // Number of success for each group and day
  int<lower=0> trials[G, N]; // Max number of possible successes
}
parameters {
  real<lower=0,upper=1> theta[G];
  real<lower=0,upper=1> p[G];
}
model {
  for (g in 1:G){
    for (n in 1:N) {
      if (y[g, n] == 0)
        increment_log_prob(log_sum_exp(bernoulli_log(1,theta[g]), 
          bernoulli_log(0,theta[g]) + binomial_log(y[g, n], trials[g, n], p[g])));
      else
        increment_log_prob(bernoulli_log(0, theta[g]) + binomial_log(y[g, n], trials[g, n], p[g]));
    }
  }
}