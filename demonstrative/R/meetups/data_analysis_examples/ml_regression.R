library(dplyr)
library(ggplot2)
library(reshape2)
library(foreach)
library(stringr)
library(lme4)
csv <- '/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/data/crime_data.csv'
data <- read.csv(csv, stringsAsFactors=F)

data <- data %>% 
  melt(id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>% 
  mutate(Year = as.numeric(str_replace(Year, 'X', '')) - 2000) %>%
  filter(!is.na(Homicide.Rate))

fit <- lmer(Homicide.Rate ~ 0 + (Year|Country), data = data)

#predict(fit, newdata = data.frame(Country='Cuba', Year=0:15))

predictions <- foreach(c=unique(d$Country), .combine=rbind) %do%{
  pred.d <- data.frame(Country=c, Year=(2014:2015)-2000)
  bootfit <- bootMer(fit, use.u = T, nsim=30, FUN=function(x) {
    predict(x, pred.d, type='response')
  })
  data.frame(Country=c, Prediction=bootfit$t[,2])
}

ordered.countries <- predictions %>% group_by(Country) %>% 
  summarise(Median=median(Prediction)) %>%
  arrange(desc(Median)) %>% .$Country

predictions %>% 
  mutate(Country=factor(as.character(Country), , levels=ordered.countries)) %>%
  ggplot(aes(x=Country, y=Prediction)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

rbind(
  data %>% filter(Country == 'British Virgin Islands'),
  predictions %>% filter(Country == 'British Virgin Islands') %>% 
    mutate(Year = 2015) %>% rename(Homicide.Rate=Prediction)
)