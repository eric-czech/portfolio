library(dplyr)
library(foreach)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')

d <- read.csv('~/data/pbto2/export/data_stan_input.csv', stringsAsFactors=F)
static.features <- c('age', 'marshall', 'gcs', 'sex')
ts.feature <- 'pbto2'
features <- c(static.features, ts.feature)

ds <- get.cleaned.data(d, features, scale=T, sample.frac=NULL, outcome.func=gos.to.binom)
dsu <- get.cleaned.data(d, features, scale=F, sample.frac=NULL, outcome.func=gos.to.binom)

unscaled.value <- function(x, var) x * sd(dsu[,var]) + mean(dsu[,var])

frac <- 1
uids <- ds$uid %>% unique
no <- length(uids)
nt <- floor(no * frac)
uids <- sample(uids, size=nt, replace=T)
dsim <- foreach(u=uids, .combine=rbind) %do% {ds %>% filter(uid==u)}

alpha <- 0
b.age <- 2
b.mar <- -.5
b.gcs <- 1
b.sex <- .01

# Rise on both sides
br <- -6; p <- .3; bc <- 0;
a1 <- br * p; a2 <- (1 - p) * br;
b1 <- 25; b2 <- -20;
c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)


# Rise on right
# br <- -5; p <- 1; bc <- 0;
# a1 <- br * p; a2 <- (1 - p) * br;
# b1 <- 25; b2 <- -20;
# c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)

# Plot ts weight function
#x <- seq(min(ds$pbto2), max(ds$pbto2), length.out = 100)
dev.off();
plot(x, double.logistic(x, a1, a2, b1, b2, c1, c2, bc), type='l')
sapply(quantile(ds$pbto2, probs=c(.1, .25, .5, .75, .99)), function(x) abline(v=x))
# unscaled.value(.2, 'pbto2')

get.w <- function(x) 
  sum(sapply(x, function(v) double.logistic(v, a1, a2, b1, b2, c1, c2, bc))) / length(x)
dpw <- dsim %>% group_by(uid) %>% 
  summarise(w=get.w(pbto2)) %>% ungroup 
  #mutate(w=scale(w))

dp <- dsim %>% inner_join(dpw, by = 'uid') %>% group_by(uid) %>% do({
  d <- .
  r1 <- alpha + d$age[1] * b.age + d$sex[1] * b.sex + d$gcs[1] * b.gcs + d$marshall[1] * b.mar
  r2 <- d$w[1] 
  p <- 1 / (1 + exp(-(r1 + r2)))
  d$outcome <- sample(0:1, prob = c(1-p, p), size=1)
  d$r1 <- r1
  d$r2 <- r2
  d$p <- p
  d
}) %>% ungroup


# hist(dp$p)
# hist(dp$r2)
# hist(dp$r1)
# hist(dp$w)


d.stan <- dp %>% select(-r1, -r2, -p, -w) %>%
  mutate(uid=as.integer(factor(uid)))

# apply(d.stan, 2, function(x)sum(is.na(x)))

d.model <- get.stan.data(d.stan, static.features, ts.feature)

setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'nonlinear_binom_2.stan'

posterior <- stan(model.file, data = d.model,
                  warmup = 100, iter = 500, thin = 10, 
                  chains = 1, verbose = FALSE)

library(parallel) # or some other parallelizing package
n.chains <- 6

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 100, iter = 3000, chains = 1, thin = 20, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)


pars <- c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha', 'p')
post <- rstan::extract(posterior)
print(posterior, pars)

rstan::traceplot(posterior, pars)
plot(posterior)

x <- seq(min(ds$pbto2), max(ds$pbto2), length.out = 100)
y.est <- get.mean.curve(post, x)
y.act <- double.logistic(x, a1, a2, b1, b2, c1, c2)
y.main <- data.frame(i=0, x=unscaled.value(x, 'pbto2'), y.est, y.act) %>% 
  melt(id.vars=c('i', 'x'), value.name = 'y')
#y.main %>% ggplot(aes(x=x, y=y, color=variable)) + geom_line()

n = length(post$lp__)
y.samp <- foreach(i=1:n, .combine=rbind) %do% {
  y <- double.logistic(x, post$a1[i], post$a2[i], post$b1[i], post$b2[i], post$c[i, 1], post$c[i, 2])
  a = log(sqrt(sum((y - y.est)^2)))
  data.frame(i, x=unscaled.value(x, 'pbto2'), y, a=a)
} %>% mutate(a=1-scale.minmax(a))

ggplot(NULL) + 
  geom_line(aes(x=x, y=y, group=variable, color=variable), size=1, data=y.main) + 
  geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
  scale_alpha(range = c(.005, .05)) + theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  xlab('PbtO2') + ylab('w(PbtO2)') + ggtitle('Timeseries Weight Function') + 
  ylim(-15, 0) + 
  ggsave('/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_right.png')

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


