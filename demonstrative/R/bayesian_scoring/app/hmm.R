
library(depmixS4)
library(TTR)
library(ggplot2)
library(dplyr)
library(foreach)

sim <- new.env()
load('../sim/sim_data_fixed.RData', envir = sim)
d <- sim$data
d$value = as.integer(d$value)

# Plot the simulated data
ggplot(d, aes(x=day, y=value, color=event)) + geom_bar(stat='identity') + facet_wrap(~group)

# Pick a group to test with
events <- c('event1', 'event2')
dt <- subset(d, group == 'Group.250' & event %in% events)
dt <- dcast(dt, day ~ event, value.var='value')


states <- paste('S', 1:2, sep='')
m1 <- depmix(list(event1 ~ 1, event2 ~ 1), 
             nstates = length(states), data = dt,
             family = list(gaussian(), gaussian()))#, 
             #respstart = rep(10, 8))

set.seed(1)
f1 <- fit(m1, verbose = FALSE)

# Initial probabilities
summary(f1, which = "prior")
# Transition probabilities
summary(f1, which = "transition")
# Reponse/emission function
summary(f1, which = "response")

# Plot the state posteriors
post <- posterior(f1)
post$day <- dt$day
post[,events] <- dt[,events]
post.plot <- melt(post[,-1], id.vars=c('day'), measure.vars=c(events, states))

ggplot() +
  geom_rect(data=subset(post.plot, variable %in% states), aes(xmin=day-28800, xmax=day+28800 , ymin=-10, ymax=-10 + 9*value, fill=variable)) +
  geom_bar(data=subset(post.plot, variable %in% events), aes(x=day, y=value, color=variable), stat='identity', fill='white') 


# Determine the predictive distribution
last.state = post[nrow(post),]
params <- unlist(sapply(f1@response, function(x){x[[1]]@parameters}))
params <- ifelse(params > 0, params, .Machine$double.xmin)
names(params) <- paste('S', 1:length(params), sep='')

grid.delta <- 1
grid <- seq(0, 1000, grid.delta)
param.dist <- foreach(s=names(params), .combine=cbind.data.frame) %do% {
  last.state[,s][1] * grid.delta * dpois(grid, params[[s]])
} %>% setNames(names(params))

n.points <- 15
ggplot(melt(param.dist[1:n.points,]), aes(x=rep(1:n.points, length(params)), y=asinh(value))) + 
  geom_line() + facet_wrap(~variable, scales="free") 

mix.dist <- apply(param.dist, 1, sum)

n.points <- 5
ggplot(data.frame(x=mix.dist[1:n.points]+1e-12), aes(x=1:n.points, y=asinh(x))) + geom_line() 

m2 <- depmix(value ~ 1, family = poisson(), nstates = 2, data = dt, 
             prior = f1@prior, transition = f1@transition, response = f1@response)
