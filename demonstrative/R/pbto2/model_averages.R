library(glmulti)
library(dplyr)


scale <- function(x) (x - mean(x)) / sd(x)
d <- read.csv('/Users/eczech/data/pbto2/export/data_modeling.csv') %>% 
  mutate_each(funs(as.numeric)) %>%
  mutate_each(funs(scale), -gos) %>%
  dplyr::select(-paco2_35_45, -icp1_0_20, -pha_7.35_7.45, -pao2_30_100, -pbto2_20_100, -starts_with('n_'))

d2c <- d %>% mutate(gos = ifelse(gos <= 3, 0, 1))

p <- d2c %>% select(-gos) %>% names
p <- c('age', 'sex', 'gcs')

options(nwarnings=1000)
glm.res.lvl.1 <- glmulti('gos', p, d2c, family='binomial', level=1)
summary(glm.res.lvl.1)
coef(glm.res.lvl.1)

tmp <- weightable(glm.res.lvl.1)
tmp <- tmp[tmp$aic <= min(tmp$aic) + 2,]
tmp

summary(glm.res.lvl.1@objects[[1]])

