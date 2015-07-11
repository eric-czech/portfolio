library(dplyr)
library(ggplot2)
library(reshape2)
library(stringr)
library(mice)
library(foreach)

csv <- '/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/data/crime_data.csv'
d <- read.csv(csv, stringsAsFactors=F)
#d <- subset(d, Country != 'Cuba')
d.m <- as.matrix(d[,2:ncol(d)])
rownames(d.m) <- d$Country
colnames(d.m) <- str_replace(colnames(d.m), 'X', '')
#d.m <- t(d.m)

m <- 10
d.imp <- mice(d.m, m=m)

complete(d.imp, 1) %>% 
  add_rownames(var='Country') %>%
  melt(id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>% 
  mutate(Year=as.numeric(str_replace(as.character(Year), 'Y.', ''))) %>% 
  ggplot(aes(x=Year, y=Homicide.Rate, color=Country)) + geom_line() + facet_wrap(~Country)

# Plots of imputations for a single country
country <- 'Cayman Islands'
sapply(1:m, function(i) unlist(complete(d.imp, i)[country,] )) %>% 
  as.data.frame %>% add_rownames(var='Year') %>% 
  melt(id.vars='Year', variable.name='Imputation', value.name='Homicide.Rate') %>% 
  mutate(Year=as.numeric(Year), Imputation=factor(as.integer(Imputation))) %>%
  ggplot(aes(x=Year, y=Homicide.Rate, color=Imputation)) + geom_line() +
  scale_x_continuous(breaks=2000:2012)

library(lme4)
# Fit models on 2000 - 2012 data
mod.fit <- foreach(i=1:m) %do% {
  d.imp.i <- complete(d.imp, i) %>% as.data.frame %>% add_rownames(var='Country') 
  d.in <- melt(d.imp.i, id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>%
    mutate(Year = as.integer(as.character(Year)), Country = as.factor(Country))
  list(model=lmer(Homicide.Rate ~ 0 + (I(Year-2000)|Country), data = d.in), data=d.in, imputation=i)
} 

# Make predictions for 2014 & 2015
mod.pred <- foreach(fit = mod.fit, .combine=rbind) %do%{
  nd <- data.frame(Country=rep(unique(fit$data$Country),16), Year=2000:2015)
  nd$Homicide.Rate <- predict(fit$model, newdata=nd, type='response')
  nd$Imputation <- fit$imputation
  nd
}

# Plot predicted values vs actual
d.all <- mod.pred %>% 
  left_join(
    d.m %>% as.data.frame %>%
      add_rownames(var='Country') %>%
      melt(id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>%
      mutate(Year = as.integer(as.character(Year)), Country = factor(Country)) %>%
      filter(!is.na(Homicide.Rate))
    , by=c('Country', 'Year')) %>% 
  rename(Homicide.Rate.Predicted=Homicide.Rate.x, Homicide.Rate.Actual=Homicide.Rate.y) 

d.all %>% 
  ggplot(aes(x=Year, y=Homicide.Rate.Predicted, color=factor(Imputation))) + 
  geom_line() + geom_point(aes(x=Year, y=Homicide.Rate.Actual), color='black') +
  facet_wrap(~Country) + theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())


# Plot 2015 predictions
d.mean <- d.all %>%
  filter(Year == 2015) %>%
  group_by(Country) %>%
  summarise(Mean.HR=mean(Homicide.Rate.Predicted)) %>% 
  arrange(desc(Mean.HR)) %>% ungroup %>% as.data.frame

d.all %>%
  filter(Year == 2015) %>%
  mutate(Country = factor(as.character(Country), levels=d.mean$Country)) %>%
  ggplot(aes(x=Country, y=Homicide.Rate.Predicted)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) + 
  theme(panel.grid.major=element_blank())
