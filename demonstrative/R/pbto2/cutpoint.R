library(dplyr)
library(ggplot2)
library(segmented)

d <- read.csv('/Users/eczech/data/ptbo2/export/cutpoint_modeling.csv')
d <- d %>% filter(!is.na(gos.3)) %>% mutate(gos.3=factor(ifelse(gos.3 >= 4, 'Good', 'Bad')))
scale <- function(x) (x - mean(x)) / sd(x)
d.scale <- d %>% mutate_each(funs(scale), one_of('marshall', 'pbto2', 'gcs', 'age', 'tsi_min'))


out.lm <- glm(gos.3 ~ pbto2 + tsi_min + marshall + gcs + age, data=d.scale, family='binomial')
m <- segmented(out.lm, seg.Z=~pbto2 + tsi_min, 
               psi=list(pbto2=-1, tsi_min=-.3), 
               control=seg.control(display=FALSE))
summary(m)

unscale <- function(var, v){
  m <- mean(d[,var])
  s <- sd(d[,var])
  v * s + m
}
unscale('pbto2', .196)
unscale('tsi_min', 3.915)