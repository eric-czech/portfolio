library(dplyr)
library(foreach)
library(qcc)

sim <- new.env()
load('sim/sim_data_fixed.RData', envir = sim)
data <- sim$data

#prior.alpha <- 100
#prior.beta <- 10000
prior.alpha <- 1
prior.beta <- 1

window.sum <- function(x, n=15, span=.2){
  res = x
  for(i in 1:length(x)){
    s <- 1
    if (i-n+1 > 1)
      s <- i-n+1
    res[i] <- sum(x[s:i], na.rm=T)
  }
  #res <- eqwma(res, lag=10)# predict(loess(y ~ x, span=span, data=data.frame(x=1:length(res), y=res), degree=1))
  ewmaSmooth(x=1:length(res), y=res, lambda=.2)$y %>% as.vector
}
window.size <- 30

posterior <- data %>%
  group_by(group, event) %>% 
  do((function(group.data){
    group <- group.data$group[1]
    event <- group.data$event[1]
    foreach(
      day=group.data$day, 
      cum.size=window.sum(group.data$size, n=window.size), 
      size=group.data$size,
      cum.value=window.sum(group.data$value, n=window.size),
      value=group.data$value,
      .combine=rbind) %do% {
        alpha <- prior.alpha + cum.value
        beta <- cum.size - cum.value + prior.beta
        #p.estimate <- alpha/ (alpha + beta)
        p.estimate <- qbeta(p=.025, shape1=alpha, shape2=beta)
        data.frame(group, event, day, value, size, fraction=value/size, p.estimate, alpha, beta)
    }
  })(.))

posterior <- 
  posterior %>% select(group, event, day, p.estimate, fraction)

id.vars <- names(posterior)[!names(posterior) %in% c('fraction', 'p.estimate')]
split <- melt(posterior, id.vars=id.vars, value.name = 'y', variable.name = 'type')

split <- split %>% 
  filter(type=='p.estimate') %>%
  group_by(group, day) %>% 
  summarise(y=sum(y), type='score', event='all') %>% 
  rbind(split)

ggplot(split, aes(x=day, y=y, color=event)) + geom_line() + 
  facet_grid(type~group, scales="free")

v <- subset(data, group=='Group4')$value
v <- data.frame(x=1:length(v), y=v)
#v$a <- EMA(v$y, ratio=.6)
#v$a <- window.sum(v$y, span=.05)#predict(loess(y ~ x, span=.05, data=v, degree=1))
v$a <- ewma(v$y)$y
v <- melt(v, id.vars='x')
ggplot(v, aes(x=x, y=value, color=variable)) + geom_line()

