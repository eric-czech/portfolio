library(plotly)
library(dplyr)
library(ggplot2)
library(reshape2)

d <- read.csv('~/data/pbto2/export/data_stan_input.csv', stringsAsFactors=F)

p <- 'pbto2'
c1 <- 14.5
c2 <- 75.1

breaks <- c(-Inf, c1, c2, Inf)

# v1 <- paste0('v <= ', c1)
# v2 <- paste0('v <= ', c2)
# v3 <- paste0('v > ', c2)

q <- d %>% group_by(uid) %>% do({
  x <- .
  v <- x %>% data.frame %>% .[,p] %>% na.omit()
  l <- cut(v, breaks = breaks, labels = c('v1', 'v2', 'v3'))
  
  replace.na <- function(x) ifelse(is.na(x), 0, x)
  data.frame(l) %>% group_by(l) %>% tally %>% 
    mutate(n=n/sum(.$n)) %>% 
    dcast(. ~ l, value.var='n') %>% select(-matches('\\.')) %>%
    mutate(gos=x$gos[1]) 
}) %>% 
  mutate_each(funs(replace.na), starts_with('v')) %>% 
  mutate(v4=v1 + v3) %>%
  mutate(gos=ifelse(gos <= 3, 0, 1))

plot_ly(q, y = v4, color = factor(gos), type = "box")

plot_ly(q2, x = v2, y = v4, type = "scatter", mode = "markers", color=factor(gos))


