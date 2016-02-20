#library(SQDA)
source('/Users/eczech/repos/portfolio/demonstrative/R/model_tests/sqda_model.R')
library(caret)
library(dplyr)

d <- twoClassSim(n=100, noiseVars = 100)
scale <- function(x)(x-mean(x))/sd(x)
X <- d %>% select(-Class) %>% mutate_each(funs(scale))
y <- d$Class

fold <- createDataPartition(y, p=.8)[[1]]
d.tr <- unname(t(X[fold,]))
colnames(d.tr) <- y[fold]
d.ts <- unname(t(X[-fold,]))
colnames(d.ts) <- y[-fold]

m <- sQDA(d.tr, d.ts, len=5)

