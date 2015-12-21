library(rstan)
library(ggplot2)

model <- '
parameters{
  real p;
}
model {
  real c;
  c <- 10;
  sinh(p/c) ~ normal(0, 2);
  increment_log_prob(log(fabs((1/c)*cosh(p/c))));
}
'

# Adding these parameters will make sampling for large normal SD's above possible
#d <- stan(model_code = model, iter=21000, chains=1, warmup = 1000, control=list(adapt_engaged=F, stepsize=5))
d <- stan(model_code = model, iter=21000, chains=1, warmup = 1000)
p <- rstan::extract(d)
# ggplot(data.frame(p=p$p), aes(x=p)) + geom_density()
# ggplot(data.frame(p=p$p), aes(x=p)) + geom_histogram()
traceplot(d)

x = 10*asinh(rnorm(20000, 0, 2))
plot(density(x))
plot(sort(x), sort(p$p))