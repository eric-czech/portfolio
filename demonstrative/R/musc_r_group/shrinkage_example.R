
library(dplyr)
library(plotly)
library(lme4)

# Actual sample data sample for group (from Scott) 
# d <- read.csv('/Users/eczech/data/meetups/musc_r_group/sampledata.csv')
# d %>% plot_ly(x=acuity, y=or_time_obs, color=factor(doctor), type='scatter', mode='markers')

# Simulated data with more samples similar to the above
n <- 25
weight.exp <- 1 # Increasing this decreases the "shrinkage" of small group means

set.seed(1)
d <- rbind(
  # Generate 3 "normal" doctors and their acuity / OR time numbers
  data.frame(doctor=1, acuity=rnorm(n, 3, .5), or_time_obs=rnorm(n, 75, 3)),
  data.frame(doctor=2, acuity=rnorm(n, 4.2, .5), or_time_obs=rnorm(n, 72, 3)),
  data.frame(doctor=3, acuity=rnorm(n, 3.2, .5), or_time_obs=rnorm(n, 82, 3)),
  
  # Generate one oddball doctor with only 2, atypical measurements
  data.frame(doctor=4, acuity=c(6.8, 5.1), or_time_obs=c(95, 82))
) %>% 
  mutate(doctor=factor(paste('Doctor', doctor))) %>%
  group_by(doctor) %>% mutate(weight=n()^(weight.exp)) %>% ungroup %>%
  mutate(weight=weight/n())

# Scatter plot function showing actual data and means for each doctor
plot.data <- function(d, d.mean){
  d %>%
    plot_ly(
      x=acuity, y=or_time_obs, color=doctor, type='scatter', mode='markers', 
      marker=list(size=5), legendgroup='Actual'
    ) %>% add_trace(
      data=d.mean, x=acuity, y=or_time_obs, color=doctor, type='scatter', mode='markers', 
      marker=list(size=15), legendgroup='Means'
    )  
}

# Plot the original simulated data and the means for each doctors stats
d.mean <- d %>% group_by(doctor) %>% summarise_each(funs(mean))
plot.data(d, d.mean)


# Create varying-intercepts models to more directly model the "average" acuity
# and or_time_obs for each doctor.  If "lm" (i.e. fixed-effects regression) was used below 
# instead of "lmer" (i.e. "random/mixed effects regression") then the results would
# be exactly equivalent to calculting the mean of each value by doctor.  The whole point
# of using "lmer" then is to get a more realistic average for each doctor if the number
# of samples is small -- where realistic means "pooling" information from the other doctors
m.ac <- lmer(acuity ~ (1|doctor), d, weights=d$weight)
m.or <- lmer(or_time_obs ~ (1|doctor), d, weights=d$weight)

# Use the model to get the predicted average acuity and or_time_obs for each doctor
d.doc <- data.frame(doctor=unique(d$doctor))
d.mean.reg <- data.frame(
  doctor=d.doc$doctor,
  acuity=predict(m.ac, d.doc),
  or_time_obs=predict(m.or, d.doc)
)
plot.data(d, d.mean.reg)

