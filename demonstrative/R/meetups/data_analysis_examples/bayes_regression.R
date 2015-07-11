
library(dplyr)
library(reshape2)
library(stringr)
library(rstan)

csv <- '/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/data/crime_data.csv'
data <- read.csv(csv, stringsAsFactors=F)


data <- data %>% 
  melt(id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>% 
  mutate(Year = as.numeric(str_replace(Year, 'X', '')) - 2000) %>%
  filter(!is.na(Homicide.Rate))


countries <- unique(data$Country)
stan.data <- list(
  N = nrow(data),
  C = length(unique(data$Country)),
  country = match(data$Country, countries),
  year = data$Year,
  homicide = data$Homicide.Rate
)

setwd('/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples')
fit = stan('bayes_regression.stan', data = stan.data , warmup = 2000, iter = 5000, thin = 100, chains = 4, verbose = FALSE)
posterior <- rstan::extract(fit)

# Convergence checks
# install.packages("ggmcmc", dependencies=TRUE) 
# library(ggmcmc) 
# param_samples <- as.data.frame(fit)[,1]
# plot(as.mcmc(param_samples))

predictions <- posterior$predictions 
colnames(predictions) <- countries
predictions <- as.data.frame(predictions) %>% 
  mutate(SampleId=1:nrow(.)) %>% 
  melt(id.vars = 'SampleId', variable.name = 'Country', value.name = 'Prediction')

ordered.countries <- predictions %>% group_by(Country) %>% 
  summarise(Median=median(Prediction)) %>%
  arrange(desc(Median)) %>% .$Country

predictions %>% 
  mutate(Country=factor(as.character(Country), , levels=ordered.countries)) %>%
  ggplot(aes(x=Country, y=Prediction)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))