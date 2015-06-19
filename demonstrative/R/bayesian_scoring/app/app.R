library(dplyr)
library(foreach)
library(qcc)


getPosterior <- function(data, input){
  
  prior.alpha <- input$prior.alpha
  prior.beta <- input$prior.beta
  
  window.sum <- function(x, n=15, lambda=.5){
    res <- x
    n <- input$window.size
    
    for(i in 1:length(x)){
      s = 1
      if (i-n+1 > 1)
        s <- i-n+1
      res[i] <- sum(x[s:i], na.rm=T)
    }
    if (!is.na(lambda))
      res <- ewmaSmooth(x=1:length(res), y=res, lambda=lambda)$y %>% as.vector
    res
  }
  
  posterior <- data %>%
    group_by(group, event) %>% 
    do((function(group.data){
      group <- group.data$group[1]
      event <- group.data$event[1]
      foreach(
        day=group.data$day, 
        cum.size=window.sum(group.data$size, n=input$window.size, lambda=(1-input$smooth.parameter)), 
        size=group.data$size,
        cum.value=window.sum(group.data$value, n=input$window.size, lambda=(1-input$smooth.parameter)),
        value=group.data$value,
        .combine=rbind) %do% {
          alpha <- prior.alpha + cum.value
          beta <- cum.size - cum.value + prior.beta
          p.estimate <- qbeta(p=input$credible.interval.p, shape1=alpha, shape2=beta)
          data.frame(group, event, day, value, size, fraction=value/size, p.estimate, alpha, beta)
        }
    })(.))
  
  posterior <- 
    posterior %>% select(group, event, day, p.estimate, fraction)
  
  id.vars <- names(posterior)[!names(posterior) %in% c('fraction', 'p.estimate')]
  split <- melt(posterior, id.vars=id.vars, value.name = 'y', variable.name = 'type')
  
  split %>% 
    filter(type=='p.estimate') %>%
    group_by(group, day) %>% 
    summarise(y=sum(y), type='score', event='all') %>% 
    rbind(split)
  

}

getMcmcPosterior <- function(data, input){
# Data format:
#   group  event        day value size
#   1 Group.10 event1 2015-01-01     0   10
#   2 Group.10 event1 2015-01-02     0   10
  n.days <- length(unique(data$day))
  offset <- 3
  trials <- dcast(data, group + event ~ day, value.var='size')[,offset:(n.days+offset-1)]
  y <- dcast(data, group + event ~ day, value.var='value')[,offset:(n.days+offset-1)]
  y <- round(y, digits=0) %>% apply(2, as.integer)
  d <- list(
    N=n.days,
    G=dim(y)[1],
    y=y,
    trials=trials
  )
  
  fit <- stan(file='../mcmc/zero-inflated-binom-set.stan', data=d, iter = 1000, chains=1)
  
  fit@sim$samples[[1]] # this returns list of parameters like theta[11] and samples for each
}

getPosteriorPlot <- function(data, input){
  ggplot(data, aes(x=day, y=y, color=event)) + geom_line() + 
    facet_grid(type~group, scales="free") 
}

getPriorPlot <- function(input){
  ggplot(data.frame(x=rbeta(10000, input$prior.alpha, input$prior.beta)), aes(x=x)) + 
    ggtitle(sprintf('Beta Prior Density\n(alpha=%s, beta=%s)', input$prior.alpha, input$prior.beta)) +
    geom_density(fill='red', alpha=.1) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
}


getEventsPlot <- function(data, input){
  data %>% 
    filter(type == 'fraction') %>%
    ggplot(aes(x=day, y=y, fill=event)) + 
    geom_bar(stat='identity') + ggtitle('Events') +
    facet_grid(type~group, scales="free") 
}

getEstimatesPlot <- function(data, input){
  data %>% 
    filter(type == 'p.estimate') %>%
    ggplot(aes(x=day, y=y, color=event)) + 
    geom_line() + ggtitle('Probability Estimates') +
    facet_grid(type~group, scales="free") 
}

getScoresPlot <- function(data, input){
  data %>% 
    filter(type == 'score') %>%
    ggplot(aes(x=day, y=y, color=event)) + 
    geom_line() + ggtitle('Scores') +
    facet_grid(type~group, scales="free") 
}

