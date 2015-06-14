library(dplyr)
library(foreach)
library(ggplot2)

source('sim/params_sim.R')

set.seed(2)


group.event.probs <- foreach(group=names(params$groups), .combine = rbind) %do% {
  data.frame(group=group, p=runif(1, .01, .05))
}

event.probs <- foreach(event=names(params$events), .combine = rbind) %do% {
  data.frame(event=event, p=runif(1, .1, .9))
}

data <- foreach(g=names(params$groups), .combine = rbind) %do% {
  foreach(e=names(params$events), .combine = rbind) %do% {
    foreach(d=params$days, i=1:length(params$days), .combine = rbind) %do% {
      
      # Get the group level zero-inflation probabilition
      p.allow.event <- subset(group.event.probs, group==g)$p
      
      group.size <- params$groups[[g]](d)
      p.event <- subset(event.probs, event==e)$p
      value <- 0
      
      # for now, the group level inflation probability is ignored and instead
      if (runif(1) < p.allow.event)
        value <- rbinom(n=1, size=group.size, prob=p.event)
      data.frame(group=g, event=e, day=d, value, size=group.size)
    }
  }
}

ggplot(data, aes(x=day, y=value, fill=event)) + geom_bar(stat='identity') + 
  facet_wrap(~group, nrow=length(names(params$groups)), scales="free")

save(data, group.event.probs, event.probs, file='sim/sim_data_variable.RData')
