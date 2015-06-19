data {
  int<lower=0> N;
  int<lower=0> y[N];
  int<lower=0> trials[N];
}
parameters {
  real<lower=0,upper=1> theta;
  real<lower=0,upper=1> p;
}
model {
  for (n in 1:N) {
    if (y[n] == 0)
      increment_log_prob(log_sum_exp(bernoulli_log(1,theta), bernoulli_log(0,theta) + binomial_log(y[n], trials[n], p)));
    else
      increment_log_prob(bernoulli_log(0,theta) + binomial_log(y[n], trials[n], p));
  }
}