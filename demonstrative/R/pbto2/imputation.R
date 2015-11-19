library(dplyr)
library(mice)
library(foreach)

csv <- '/Users/eczech/data/ptbo2/export/data_to_impute.csv'
factor.gos <- function(x) factor(x, levels=1:5, ordered = T)
d.raw <- read.csv(csv, stringsAsFactors=F) %>%
  #select(matches('gos'), uid) %>%
  mutate_each(funs(factor.gos), matches('gos'))
d <- d.raw %>% dplyr::select(-uid)


#d.t <- d[!is.na(d['gos.3']) & !is.na(d['marshall']),]
#d.t <- d[!is.na(d['gos.3']),]
d.m <- m.polr <- d %>% 
  dplyr::select(age, marshall, gcs, pbto2_mean_under_15, pao2_p990, gos.3) %>%
  na.omit 
m.polr <- polr(gos.3 ~ ., d.m, Hess=TRUE)
summary(m.polr)

m.polr <- with(d.imp, )
summary(pool(m.polr))

m = 10
d.imp <- mice(d, m=10)

d %>% add_rownames %>% 
  left_join(d.imp$imp$gos.3 %>% setNames(paste('i', 1:m, sep='.')) %>% add_rownames, by='rowname') %>% 
  filter(!is.na(i.1)) %>% dplyr::select(matches('i\\.'), matches('gos\\.'))


library(MASS)
m.polr <- with(d.imp, polr(gos.3 ~ age + marshall + gcs + pbto2_mean_under_15, Hess=TRUE))
summary(pool(m.polr))

complete(d.imp, 1) %>% dplyr::select(pao2_std, gos.3) %>% ggplot(aes(x=pao2_std, y=gos.3)) + geom_point()

mod.fit <- foreach(i=1:m) %do% {
  d.imp.i <- complete(d.imp, i) %>% as.data.frame %>% add_rownames(var='Country') 
  d.in <- melt(d.imp.i, id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>%
    mutate(Year = as.integer(as.character(Year)), Country = as.factor(Country))
  list(model=lmer(Homicide.Rate ~ 0 + (I(Year-2000)|Country), data = d.in), data=d.in, imputation=i)
} 

d.polr <- d %>% mutate(outcome=factor(outcome, levels=1:5, labels=c('Dead', 'Bad1', 'Bad2', 'Good1', 'Good2'))) 
m.polr <- polr(outcome ~ pbto2_p5 + age + marshall + gcs + sex, data = d.polr, Hess=TRUE)
coefs <- coef(summary(m.polr))
p <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
coefs <- cbind(coefs, "p value" = p)
coefs
