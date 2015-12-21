
library(foreach)
library(dplyr)

get.sim.data.from.actual <- function(ds, a1, a2, b1, b2, c1, c2, ts.feature='pbto2'){
  alpha <- 1
  b.age <- 2
  b.mar <- -.5
  b.gcs <- 1
  b.sex <- .01
  
  get.w <- function(x) 
    sum(sapply(x, function(v) double.logistic(v, a1, a2, b1, b2, c1, c2, bc))) / length(x)
  ds$value = ds[,ts.feature]
  ds <- ds %>% select(-matches(ts.feature))
  dpw <- ds %>% group_by(uid) %>% 
    summarise(w=get.w(value)) %>% ungroup
  
  ds %>% inner_join(dpw, by = 'uid') %>% group_by(uid) %>% do({
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
}


# d <- read.csv('~/data/pbto2/export/data_stan_input.csv', stringsAsFactors=F)
# ct <- d %>% group_by(uid) %>% summarise(n=length(pbto2[!is.na(pbto2)])) %>% 
#   .$n %>% table %>% data.frame %>% setNames(c('n', 'ct'))
# paste(ct$n, collapse=",")
# d$pbto2 %>% hist(breaks=50)
# hist(v.samp, breaks=50)

# ts.v <- rlnorm(10000, 3.5, .5)-10
# ts.v <- ifelse(ts.v < 0, 0, ts.v)
# v.samp <- scale(ts.v)

get.emp.dist <- function(d, var){
  v <- d[,var]
  v <- v[!is.na(v)]
  v <- scale(v)
  dist <- data.frame(v) %>% group_by(v) %>% tally() %>% setNames(c('v', 'ct')) %>% ungroup
  dist$ct <- dist$ct / sum(dist$ct)
  dist
}

get.sim.data <- function(d, a1, a2, b1, b2, c1, c2, seed=123, n=100, ts.feature='pbto2'){
  set.seed(seed)
  
  ts.dist <- d %>% dplyr::select_(.dots=c('uid', ts.feature)) %>% na.omit() %>% setNames(c('uid', 'value'))
  ts.value.unscaled <- ts.dist$value
  ts.value.scaled <- scale(ts.value.unscaled)
  ts.dist <- ts.dist %>% mutate(value=scale(value)) %>%
    group_by(uid) %>% summarise(m=mean(value), s=sd(value), n=length(value))

  alpha <- 1
  b.age <- 2
  b.mar <- -.5
  b.gcs <- 1
  b.sex <- .01
#   alpha <- 2
#   b.age <- 0
#   b.mar <- 0
#   b.gcs <- 0
#   b.sex <- 0
  
  age.dist <- get.emp.dist(d, 'age')
  mar.dist <- get.emp.dist(d, 'marshall')
  gcs.dist <- get.emp.dist(d, 'gcs')
  sex.dist <- get.emp.dist(d, 'sex')
  
  get.w <- function(x) 
    sum(sapply(x, function(v) double.logistic(v, a1, a2, b1, b2, c1, c2))) / length(x)
  res <- foreach(i=1:n, .combine=rbind) %do% {
    age <- sample(age.dist$v, prob=age.dist$ct, size=1)
    gcs <- sample(gcs.dist$v, prob=gcs.dist$ct, size=1)
    marshall <- sample(mar.dist$v, prob=mar.dist$ct, size=1)
    sex <- sample(sex.dist$v, prob=sex.dist$ct, size=1)
    r1 <- alpha + age * b.age + sex * b.sex + gcs * b.gcs + marshall * b.mar
    
    ts.id <- sample(1:nrow(ts.dist), size=1)
    ts.rec <- ts.dist[ts.id,]
    v.ts <- rnorm(n = ts.rec$n[1], mean = ts.rec$m[1], sd = ts.rec$s[1])
    v.ts <- ifelse(v.ts < min(ts.value.scaled), min(ts.value.scaled), v.ts)
    v.ts <- ifelse(v.ts > max(ts.value.scaled), max(ts.value.scaled), v.ts)
    r2 <- get.w(v.ts)
    
    p <- 1 / (1 + exp(-(r1 + r2)))
    if (is.na(p))
      browser()
    outcome <- sample(0:1, prob = c(1-p, p), size=1)
    
    data.frame(age, gcs, marshall, sex, outcome, uid=i, r1, r2, p, value=v.ts)
  }
  list(res=res, ts.value.unscaled=ts.value.unscaled)
}

