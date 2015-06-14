library(dplyr)
library(lubridate)

# Create namespace for parameters
params <- new.env()

# Group name and number of members within each group -- each group is presumed to consist
# of individual members where the simulation will to do the following to
# create an overall event count for a group on a given day:
# 1. Set a probability that each member within the group will have an event (as a bernoulli trial)
# 2. Simulate the occurrence of the event for each member
# 3. Sum up the number of members for which the event occurred (producing a binomially distributed count)
params$groups <- list(
  Group.10=function(d) 10, 
  Group.50=function(d) 50, 
  Group.100=function(d) 100
)

# Vector of event names and corresponding weights; these weights are
# used to determine how much each type of event contributes to the overall score
params$events <- c(Event1=.1, Event2=.5, Event3=.8)
# Normalize the event weights as a fraction of the overall weight sum; 
# i.e. make sure the weights sum to 1
params$events <- events / sum(events)

# Date range to run simulation for
params$days <- seq(ymd('2015-01-01'), ymd('2015-04-31'), by = 'days')






