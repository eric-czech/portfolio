library(dplyr)
library(foreach)
library(rstan)
library(dplyr)
library(ggplot2)
library(reshape2)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/sim/data_gen.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

static.features <- c('age', 'marshall', 'gcs', 'sex')
ts.feature <- 'pbto2'
features <- c(static.features, ts.feature)

# Upward bend
# br <- -6; p <- .3; bc <- 0;
# a1 <- br * p; a2 <- (1 - p) * br;
# b1 <- 25; b2 <- -20;
# c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)

# Down on left only
# br <- -6; p <- 1; bc <- 0;
# a1 <- br * p; a2 <- (1 - p) * br;
# b1 <- 3; b2 <- 0;
# c1 <- -.1; c2 <- 1; # set based on quantiles (25/75%)

# Upward bend
br <- -4; p <- .3; bc <- 0;
a1 <- br * p; a2 <- (1 - p) * br;
b1 <- -25; b2 <- 20;
c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)

# Staircase 
# br <- -6; p <- .3; bc <- 0;
# a1 <- br * p; a2 <- (1 - p) * br;
# b1 <- 12; b2 <- 12;
# c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)

# No effect
# br <- 0; p <- .3; bc <- 0;
# a1 <- br * p; a2 <- (1 - p) * br;
# b1 <- 25; b2 <- -20;
# c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)


# Use transformed actual data
du <- get.long.data(features, scale.vars=F, outcome.func=gos.to.binom, reset.uid=T)
ds <- du %>% mutate_each_(funs(scale), features)

n.uid <- length(unique(ds$uid))
v <- du[,ts.feature]
unscaled.value <- function(x) x * sd(v) + mean(v)
x <- seq(min(ds[,ts.feature]), max(ds[,ts.feature]), length.out = 100)
dp <- get.sim.data.from.actual(ds, a1, a2, b1, b2, c1, c2)
d.stan <- dp %>% select(-r1, -r2, -p, -w) %>%
  mutate(uid=as.integer(factor(uid)))

# Use simulated data
# n.uid <- 250
# sim.data <- get.sim.data(d, a1, a2, b1, b2, c1, c2, n=n.uid, seed=123, ts.feature=ts.feature) 
# dp <- sim.data$res %>% mutate_each_(funs(scale), static.features)
# v <- sim.data$ts.value.unscaled
# x <- seq(min(dp$value), max(dp$value), length.out = 100)
# unscaled.value <- function(x) x * sd(v) + mean(v)
# d.stan <- dp %>% select(-r1, -r2, -p) %>%
#   mutate(uid=as.integer(factor(uid)))

# Diagnostics
plot(x, double.logistic(x, a1, a2, b1, b2, c1, c2, bc), type='l')
sapply(quantile(dp$value, probs=c(.1, .25, .5, .75, .99)), function(x) abline(v=x))

dp %>% group_by(uid) %>% summarise(r2=min(r2), o=min(outcome), p=min(p)) %>% 
  ungroup %>% arrange(r2) %>% melt(id.vars=c('uid', 'o')) %>% 
  ggplot(aes(x=value, color=factor(o))) + geom_density() + facet_wrap(~variable, scales='free')
# hist(dp$p)
# hist(dp$r2)
# hist(dp$r1)
# hist(dp$outcome)


# Run model

d.model <- get.stan.data(d.stan, static.features, 'value')

setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'nonlinear_binom_2.stan'

posterior <- stan(model.file, data = d.model,
                  warmup = 100, iter = 1000, thin = 2, 
                  chains = 1, verbose = FALSE)
post <- rstan::extract(posterior)

# adapt_delta

library(parallel) # or some other parallelizing package
n.chains <- 4

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 300, iter = 1200, chains = 1, thin = 5, 
       verbose = FALSE, chain_id=chain)
}) 
posterior <- sflist2stanfit(posterior)
post <- rstan::extract(posterior)


pars <- c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha', 'p')
print(posterior, pars)

rstan::traceplot(posterior, pars)
plot(posterior)
pairs(data.frame(post$b1, post$b2, post$betaz, post$alpha, post$c1, post$c2))


y.est <- get.mean.curve(post, x, agg.func=median)
y.act <- double.logistic(x, a1, a2, b1, b2, c1, c2)
y.main <- data.frame(i=0, x=unscaled.value(x), y.est, y.act) %>% 
  melt(id.vars=c('i', 'x'), value.name = 'y')
#y.main %>% ggplot(aes(x=x, y=y, color=variable)) + geom_line()

n = length(post$lp__)
y.samp <- foreach(i=1:n, .combine=rbind) %do% {
  y <- double.logistic(x, post$a1[i], post$a2[i], post$b1[i], post$b2[i], post$c[i, 1], post$c[i, 2])
  a = log(sqrt(sum((y - y.est)^2)))
  data.frame(i, x=unscaled.value(x), y, a=a)
} %>% mutate(a=1-scale.minmax(a))

v.hist <- hist(v, plot=F, breaks=length(x))
min.v <- min(min(y.main$y), min(y.samp$y))
max.v <- max(max(y.main$y), max(y.samp$y))
v.hist <- data.frame(x=v.hist$mids, y=v.hist$counts/sum(v.hist$counts))
v.hist$y = min.v + .15 * abs(max.v - min.v) * scale.minmax(v.hist$y)

ggplot(NULL) + 
  geom_line(aes(x=x, y=y, group=variable, color=variable), size=1, data=y.main) + 
  geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
  scale_alpha(range = c(.05, .05)) + theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  xlab('PbtO2') + ylab('w(PbtO2)') + ggtitle(paste0('Timeseries Weight Function (n= ', n.uid, ')')) + 
  xlim(0, 100) + 
  geom_rect(aes(xmax=x+.5, xmin=x-.5, ymin=min.v, ymax=y), data=v.hist, alpha=.3) +
  ggsave('~/repos/portfolio/demonstrative/R/pbto2/sim/images/act_downward.png')

# bins = seq(0, max(fun.sim$y)+1, length.out = 50)
# #bins = seq(0, 100, length.out = 500)
# d.hist <- fun.sim %>% group_by(x) %>% do({
#   h <- hist(.$y, breaks=bins, plot=F)
#   data.frame(x=.$x[1], d=h$density, y=bins[1:(length(bins)-1)])
# })
# d.hist %>% ggplot(aes(x=x, y=y, fill=d)) + geom_tile() + scale_fill_gradient(trans = "log")
# 
# fun.sim %>% ggplot(aes(x=x, y=y, color=i, group=i)) + geom_line() + scale_y_sqrt()
# fun.sim %>% ggplot(aes(x=factor(x), y=y)) + geom_boxplot() + scale_y_sqrt()


