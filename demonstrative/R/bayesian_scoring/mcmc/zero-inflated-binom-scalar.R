library(rstan)
library(foreach)
library(coda)

p0 <- .2
pN <- .5
trials <- 100

y <- foreach(i=1:1000, .combine=c) %do% {
  if (runif(1) < p0)
    0
  else
    rbinom(n = 1, size = trials, p = pN)
    #rpois(1, lambda)
}


data <- list(
  y=y,
  N=length(y),
  trials=rep(trials, length(y))
)
fit <- stan(file='mcmc/zero-inflated-binom-scalar.stan', data=data, iter = 1000, chains=2)

stan2coda <- function(fit) {
  mcmc.list(lapply(1:ncol(fit), function(x) mcmc(as.array(fit)[,x,])))
}
plot(stan2coda(fit))