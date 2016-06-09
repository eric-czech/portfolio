
library(dplyr)
library(plotly)
library(lme4)
library(reshape2)
library(blme)
d <- read.csv('/Users/eczech/data/meetups/musc_r_group/sampledata.csv')

d %>%
  plot_ly(x=acuity, y=or_time_obs, color=factor(doctor), type='scatter', mode='markers')

n <- 25
set.seed(1)
d <- rbind(
  data.frame(doctor=1, acuity=rnorm(n, 3, .5), or_time_obs=rnorm(n, 75, 3)),
  data.frame(doctor=2, acuity=rnorm(n, 4.2, .5), or_time_obs=rnorm(n, 72, 3)),
  data.frame(doctor=3, acuity=rnorm(n, 3.2, .5), or_time_obs=rnorm(n, 82, 3)),
  data.frame(doctor=4, acuity=c(6.8, 5.1), or_time_obs=c(95, 82))
  #data.frame(doctor=4, acuity=c(8), or_time_obs=c(100))
) %>% 
  mutate(doctor=factor(paste('Doctor', doctor))) %>%
  group_by(doctor) %>% mutate(weight=n()^(1/2)) %>% ungroup %>%
  mutate(weight=weight/n())


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
d.mean <- d %>% group_by(doctor) %>% summarise_each(funs(mean))
plot.data(d, d.mean)


m.ac <- lmer(acuity ~ (1|doctor), d, weights=d$weight)
m.or <- lmer(or_time_obs ~ (1|doctor), d, weights=d$weight)
d.doc <- data.frame(doctor=unique(d$doctor))
d.mean.reg <- data.frame(
  doctor=d.doc$doctor,
  acuity=predict(m.ac, d.doc),
  or_time_obs=predict(m.or, d.doc)
)
plot.data(d, d.mean.reg)

