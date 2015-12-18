library(foreach)
library(dplyr)

get.first.row <- function(d){
  foreach(u=unique(d$uid), .combine=rbind) %do% {subset(d, uid == u) %>% head(1)}
}

get.stan.data <- function(d.stan, static.features, ts.feature, n.outcome=3){
  d.stan <- data.frame(d.stan)
  d.stan.uid <- get.first.row(d.stan)
  list(
    N_OUTCOME = n.outcome,
    N_OBS = nrow(d.stan),
    N_VARS = length(static.features),
    N_UID = max(d.stan$uid),
    y = d.stan.uid %>% .$outcome %>% as.integer,
    x = d.stan.uid[,static.features],
    z = d.stan[,ts.feature],
    uid = d.stan$uid,
    min_z = min(d.stan[,ts.feature]),
    max_z = max(d.stan[,ts.feature])
  )
}

get.stan.data.cv <- function(d.tr, d.ho, static.features, ts.feature, n.outcome=3){
  d.tr <- data.frame(d.tr)
  d.ho <- data.frame(d.ho)
  d.uid.tr <- get.first.row(d.tr)
  d.uid.ho <- get.first.row(d.ho)
  list(
    N_OUTCOME = n.outcome,
    N_OBS = nrow(d.tr),
    N_OBS_HO = nrow(d.ho),
    N_VARS = length(static.features),
    N_UID = max(d.tr$uid),
    N_UID_HO = max(d.ho$uid),
    y = d.uid.tr %>% .$outcome %>% as.integer,
    y_ho = d.uid.ho %>% .$outcome %>% as.integer,
    x = d.uid.tr[,static.features],
    x_ho = d.uid.ho[,static.features],
    z = d.tr[,ts.feature],
    z_ho = d.ho[,ts.feature],
    uid = d.tr$uid,
    uid_ho = d.ho$uid,
    min_z = min(d.tr[,ts.feature]),
    max_z = max(d.tr[,ts.feature])
  )
}


double.logistic <- function(x, a1, a2, b1, b2, c1, c2, c=0){
  r1 <- a1 / (1 + exp(b1 * (x - c1)))
  r2 <- a2 / (1 + exp(b2 * (x - c2)))
  #.5 * (r1 + r2)
  c + r1 + r2
}

single.logistic <- function(x, a, b, c){
  a / (1 + exp(b * (x - c)))
}

get.mean.curve <- function(post, x, agg.func=median){
  a1 = agg.func(post$a1)
  a2 = agg.func(post$a2)
  b1 = agg.func(post$b1)
  b2 = agg.func(post$b2)
  c1 = agg.func(post$c[,1])
  c2 = agg.func(post$c[,2])
  double.logistic(x, a1, a2, b1, b2, c1, c2)
}

get.slogit.mean.curve <- function(post, x, agg.func=median){
  a = agg.func(post$betaz)
  b = agg.func(post$b)
  c = agg.func(post$c)
  single.logistic(x, a, b, c)
}


# plot.posterior.functions <- function(post, unscaled.value, ){
#   
# 
# n = length(post$lp__)
# y.samp <- foreach(i=1:n, .combine=rbind) %do% {
#   y <- double.logistic(x, post$a1[i], post$a2[i], post$b1[i], post$b2[i], post$c[i, 1], post$c[i, 2])
#   a = log(sqrt(sum((y - y.est)^2)))
#   data.frame(i, x=unscaled.value(x), y, a=a)
# } %>% mutate(a=1-scale.minmax(a))
# 
# v.hist <- hist(v, plot=F, breaks=length(x))
# min.v <- min(min(y.main$y), min(y.samp$y))
# max.v <- max(max(y.main$y), max(y.samp$y))
# v.hist <- data.frame(x=v.hist$mids, y=v.hist$counts/sum(v.hist$counts))
# v.hist$y = min.v + .15 * abs(max.v - min.v) * scale.minmax(v.hist$y)
# 
# ggplot(NULL) + 
#   geom_line(aes(x=x, y=y, group=variable, color=variable), size=1, data=y.main) + 
#   geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
#   scale_alpha(range = c(.01, .05)) + theme_bw() +
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
#   xlab('PbtO2') + ylab('w(PbtO2)') + ggtitle('Timeseries Weight Function') + 
#   #xlim(-10, 100) + 
#   geom_rect(aes(xmax=x+.5, xmin=x-.5, ymin=min.v, ymax=y), data=v.hist, alpha=.3) +
#   ggsave('~/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1.png')
# }
# 
