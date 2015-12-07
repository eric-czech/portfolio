library(rstan)
model <- '
parameters{
  real p;
}
model {
  p ~ double_exponential(0, 10);
}
'

d <- stan(model_code = model, iter=10000)
p <- rstan::extract(d)
hist(p$p)