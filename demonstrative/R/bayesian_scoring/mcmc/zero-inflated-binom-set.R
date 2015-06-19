library(rstan)
library(foreach)
library(coda)

n.days <- 1000
groups <- list(
  group1=c(100, .8, .5),
  group2=c(125, .8, .5),
  group3=c(200, .8, .5)
)

d <- foreach(g=names(groups), .combine=rbind) %do% {
    foreach(i=1:n.days, .combine=rbind) %do% {
      v <- 0
      if (runif(1) >= groups[[g]][2])
        v <- rbinom(n = 1, size = groups[[g]][1], p = groups[[g]][3])
      data.frame(group=g, day=i, value=v)
    }
  }

trials <- foreach(g=names(groups), .combine=rbind) %do% {
  rep(groups[[g]][1], n.days)
}

data <- list(
  N=n.days,
  G=length(groups),
  y=dcast(d, group ~ day)[,1:n.days],
  trials=trials
)

fit <- stan(file='mcmc/zero-inflated-binom-set.stan', data=data, iter = 1000, chains=2)

stan2coda <- function(fit) {
  mcmc.list(lapply(1:ncol(fit), function(x) mcmc(as.array(fit)[,x,])))
}
plot(stan2coda(fit))