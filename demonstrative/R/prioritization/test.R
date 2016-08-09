
library(dplyr)
library(plotly)
library(ggplot2)
library(reshape2)

d <- mtcars
names(d) <- sapply(names(d), toupper)

wts <- c(
  mpg = 50,
  hp = 50,
  gear = 50
)
wts <- wts / sum(wts)


scale <- function(x) (x - min(x)) / (max(x) - min(x))
d.score <- d %>% 
  add_rownames(var='Vehicle.Make') %>%
  select(Vehicle.Make, MPG, HP, GEAR) %>%
  mutate_each(funs(scale), -Vehicle.Make) %>%
  mutate(Priority.Score=wts['mpg'] * MPG + wts['hp'] * HP + wts['gear'] * GEAR)

d.score %>% 
  arrange(desc(Priority.Score)) %>% head(10) %>%
  melt(id.vars=c('Vehicle.Make', 'Priority.Score'), value.name='Value', variable.name='Variable') %>%
  arrange(Priority.Score) %>%
  plot_ly(
    y=Vehicle.Make, x=Value, color=Variable, 
    type='bar', orientation='h', 
    text = paste("Score =", round(Priority.Score, 2))
  ) %>%
  layout(barmode='stack', margin=list(l=150, b=120))

d.score %>% 
  mutate(MPG = MPG * wts['mpg'], HP = HP * wts['mpg'], GEAR = GEAR * wts['gear']) %>%
  select(-Priority.Score) %>%
  melt(id.vars=c('Vehicle.Make'), value.name='Value', variable.name='Variable') %>%
  plot_ly(x=Value, color=Variable, type='histogram', opacity=.8) %>%
  layout(title='Variable Contribution Distribution')