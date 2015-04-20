library(dplyr)
library(foreach)
library(ggplot2)

source('sim/params_sim.R')

set.seed(2)


group.event.probs <- data.frame(group=names(params$groups), p=.1)

event.probs <- foreach(event=names(params$events), .combine = rbind) %do% {
  data.frame(event=event, p=runif(1, .1, .9))
}

event.dist <- 1:length(params$days) / length(params$days)
event.dist <- dbeta(event.dist, 3, 3)
event.dist <- event.dist * (1 / max(event.dist))

event.fractions <- foreach(d=params$days, i=1:length(params$days), .combine = rbind) %do% {
  foreach(e=names(params$events), .combine = rbind) %do% {
    p.event <- subset(event.probs, event==e)$p
    p.allow.event <- .1 * event.dist[[i]]
    fraction <- 0
    if (runif(1) < p.allow.event)
      fraction <- rbinom(n=1, size=100, prob=.1) / 100
    data.frame(event=e, day=d, fraction)
  }
}

data <- foreach(g=names(params$groups), .combine = rbind) %do% {
  foreach(e=names(params$events), .combine = rbind) %do% {
    foreach(d=params$days, i=1:length(params$days), .combine = rbind) %do% {
      fraction = subset(event.fractions, event==e & day==d)$fraction[1]
      group.size <- params$groups[[g]](d)
      data.frame(group=g, event=e, day=d, value=fraction * group.size, size=group.size)
    }
  }
}

ggplot(data, aes(x=day, y=value, fill=event)) + geom_bar(stat='identity') + 
  facet_wrap(~group, nrow=length(names(params$groups)), scales="free")

save(data, group.event.probs, event.probs, file='sim/sim_data_fixed.RData')
