library(dplyr)
library(rstan)
library(ggplot2)

### Create some raw data

# These are the values the model should be able to recover
engineer.slope <- -1
lunch.lady.slope <- 1
sales.guy.slope <- 2


logit <- function(x){
  log(x / (1 - x))
}
invlogit <- function(x){
  value <- exp(x)/(1+exp(x))
  return(value)
}

# Create the actual raw data ... err this part is kind of ugly
# but the data it creates looks less confusing (shown right after)
xn <- 100
#x <- rnorm(n=xn, mean=0, sd=1)
x <- sample(seq(-2, 2, .001), size = xn, replace = T)
data <- data.frame(
  linear.response.1 = c(
    engineer.slope * x + logit(runif(xn)),  
    lunch.lady.slope * x + logit(runif(xn)),
    sales.guy.slope * x + logit(runif(xn))
  ),
  title = c(
    rep('Engineer', xn), 
    rep('Lunch Lady', xn),
    rep('Sales Guy', xn)
  )) %>% 
  group_by(title) %>% 
  do({ 
    data.frame(
      performance=cut(invlogit(.$linear.response.1), breaks = c(0, .33, .66, 1), labels = c('C', 'B', 'A')),
      response.2 = sample(seq(-2, 2, .001), size = xn, replace = T),
      response.1 = x,
      linear.response.1 = .$linear.response.1
    )
  }) 


# Raw data looks like:

# > head(data)
#      title performance response.2 response.1 -> This is some kind of "personality" trait
# 1 Engineer           B         10 -1.4601307
# 2 Engineer           C          9 -4.1381546
# 3 Engineer           A          8  3.0025285
# 4 Engineer           B         11  2.8523501
# 5 Engineer           B         10  1.3669751
# 6 Engineer           A          8  0.5005971

# The 'response.1' field goes up as the performance goes up,
# but at a different rate for each title:
#
# * and 'response.2' is just some random value not related to anything
# 
#        title performance avg_response.1 avg_response.2
# 1   Engineer           C          -1.46           9.39
# 2   Engineer           B          -0.62           9.81
# 3   Engineer           A           1.14           9.19
# 4 Lunch Lady           C          -2.29           9.23
# 5 Lunch Lady           B           0.97           8.62
# 6 Lunch Lady           A           2.11           9.50
# 7  Sales Guy           C          -2.32          10.19
# 8  Sales Guy           B          -0.58          10.00
# 9  Sales Guy           A           2.83           9.06



### Create a 'Performance' model based on employee personality traits, 
### with a hierarchical separation for each job title

stan.model <- '
  data {
    int<lower=2> N_PERF_VALS;                // # of possible performance measures like "A" or "B", etc.
    int<lower=0> N_OBS;                      // # of employees in question (i.e. number of observations)
    int<lower=1> N_VARS;                     // # of personality traits
    int<lower=1> N_JOBS;                     // # of different titles possible
    int<lower=1,upper=N_PERF_VALS> y[N_OBS]; // Vector of performance evaluations
    int<lower=1> g[N_OBS];                   // "Group" or rather "Job title" for each employee
    row_vector[N_VARS] x[N_OBS];             // Matrix of personality traits for each employee
  }
  parameters {
    vector[N_VARS] beta[N_JOBS];   // Slope estimates by job title
    ordered[N_PERF_VALS-1] cutpoints; 
  }
  model {
    // Assign normal priors to coefficients to estimate
    for (i in 1:N_JOBS)
      beta[i]  ~ normal(0,5);

    // Create model where the output is predicted by the per-job-title 
    // intercept plus the per-job-title slope times each personality trait
    for (i in 1:N_OBS)
      y[i] ~ ordered_logistic(x[i] * beta[g[i]], cutpoints);
  }
'

# Convert the raw data created into Stan data
d <- list(
  N_PERF_VALS = length(levels(data$performance)), # One for each performance grade (A, B, or C)
  N_OBS = nrow(data),
  N_VARS = 2,  # One for each of the two personality 'responses'
  N_JOBS = 3,  # One for each of the job titles
  y = as.integer(data$performance),
  x = data[,c('response.1', 'response.2')],
  g = as.integer(data$title)
)

# Run the model and check that the coefficients line up with what was expected
fit <- stan(model_code = stan.model, data = d, warmup = 100, iter = 1000, thin = 5, chains = 1, verbose = FALSE)

params = list(
  'Engineer.Response.1.Beta'=list(n='beta[1,1]', actual=engineer.slope, r=1),
  'Engineer.Response.2.Beta'=list(n='beta[1,2]', actual=0, r=2),
  'Lunch.Lady.Response.1.Beta'=list(n='beta[2,1]', actual=lunch.lady.slope, r=1),
  'Lunch.Lady.Response.2.Beta'=list(n='beta[2,2]', actual=0, r=2),
  'Sales.Guy.Response.1.Beta'=list(n='beta[3,1]', actual=sales.guy.slope, r=1),
  'Sales.Guy.Response.2.Beta'=list(n='beta[3,2]', actual=0, r=2)
)

samples <- foreach(pn=names(params), i=icount(), .combine=rbind) %do%{
  param = params[[pn]]
  data.frame(param=pn, title=str_split(pn, '\\.')[[1]][1], 
             sample=fit@sim$samples[[1]][[ param[['n']] ]], 
             actual=param[['actual']], response=paste('Slope for Response', param[['r']]))
}
vlines <- samples %>% group_by(title, param, response) %>% summarise(actual=actual[1])

ggplot(samples, aes(x=sample)) + geom_density() + 
  facet_grid(title~response) + 
  geom_vline(aes(xintercept=actual), color='red', data = vlines) + theme_bw()


