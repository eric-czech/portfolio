
library(foreach)
library(dplyr)

get.sim.data.from.actual <- function(ds){
  alpha <- 1
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
  # x <- seq(min(ds[,ts.feature]), max(ds[,ts.feature]), length.out = 100)
  # dev.off();
  # plot(x, double.logistic(x, a1, a2, b1, b2, c1, c2, bc), type='l')
  # sapply(quantile(ds$pbto2, probs=c(.1, .25, .5, .75, .99)), function(x) abline(v=x))
  # unscaled.value(.2, 'pbto2')
  
  get.w <- function(x) 
    sum(sapply(x, function(v) double.logistic(v, a1, a2, b1, b2, c1, c2, bc))) / length(x)
  dpw <- du %>% group_by(uid) %>% 
    summarise(w=get.w(pbto2)) %>% ungroup 
  #mutate(w=scale(w))
  
  du %>% inner_join(dpw, by = 'uid') %>% group_by(uid) %>% do({
    d <- .
    r1 <- alpha + d$age[1] * b.age + d$sex[1] * b.sex + d$gcs[1] * b.gcs + d$marshall[1] * b.mar
    #r1 <- alpha + d$age[1] * b.age
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
}


# d <- read.csv('~/data/pbto2/export/data_stan_input.csv', stringsAsFactors=F)
# ct <- d %>% group_by(uid) %>% summarise(n=length(pbto2[!is.na(pbto2)])) %>% 
#   .$n %>% table %>% data.frame %>% setNames(c('n', 'ct'))
# paste(ct$n, collapse=",")
# d$pbto2 %>% hist(breaks=50)
# hist(v.samp, breaks=50)

get.emp.dist <- function(d, var){
  v <- d[,var]
  v <- v[!is.na(v)]
  v <- scale(v)
  dist <- data.frame(v) %>% group_by(v) %>% tally() %>% setNames(c('v', 'ct')) %>% ungroup
  dist$ct <- dist$ct / sum(dist$ct)
  dist
}

get.sim.data <- function(d, n=100, seed=123){
  set.seed(123)
  
  ts.n.dist <- d %>% group_by(uid) %>% summarise(v=length(pbto2)) %>% 
    group_by(v) %>% tally %>% setNames(c('v', 'ct'))
  ts.n.dist$ct <- ts.n.dist$ct / sum(ts.n.dist$ct)
  
  v.samp <- rlnorm(10000, 3.5, .5)-10
  v.samp <- ifelse(v.samp < 0, 0, v.samp)
  v.samp <- scale(v.samp)
  
  alpha <- 1
  b.age <- 2
  b.mar <- -.5
  b.gcs <- 1
  b.sex <- .01
  
  br <- -6; p <- .3; bc <- 0;
  a1 <- br * p; a2 <- (1 - p) * br;
  b1 <- 25; b2 <- -20;
  c1 <- -.6; c2 <- .55; # set based on quantiles (25/75%)
  
  age.dist <- get.emp.dist(d, 'age')
  mar.dist <- get.emp.dist(d, 'marshall')
  gcs.dist <- get.emp.dist(d, 'gcs')
  sex.dist <- get.emp.dist(d, 'sex')
  
  get.w <- function(x) 
    sum(sapply(x, function(v) double.logistic(v, a1, a2, b1, b2, c1, c2, bc))) / length(x)
  foreach(i=1:n, .combine=rbind) %do% {
    age <- sample(age.dist$v, prob=age.dist$ct, size=1)
    gcs <- sample(gcs.dist$v, prob=gcs.dist$ct, size=1)
    marshall <- sample(mar.dist$v, prob=mar.dist$ct, size=1)
    sex <- sample(sex.dist$v, prob=sex.dist$ct, size=1)
    r1 <- alpha + age * b.age + sex * b.sex + gcs * b.gcs + marshall * b.mar
    n.ts <- sample(ts.n.dist$v, prob=ts.n.dist$ct, size=1)
    v.ts <- sample(v.samp, size=n.ts)
    r2 <- get.w(v.ts)
    p <- 1 / (1 + exp(-(r1 + r2)))
    if (is.na(p))
      browser()
    outcome <- sample(0:1, prob = c(1-p, p), size=1)
    
    data.frame(age, gcs, marshall, sex, outcome, uid=i, r1, r2, p, pbto2=v.ts)
  }
}

