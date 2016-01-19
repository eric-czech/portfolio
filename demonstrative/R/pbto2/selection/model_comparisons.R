library(glmulti)
library(dplyr)
library(cvTools)
library(MASS)
library(MLmetrics)
library(dummies)
library(foreach)
library(plotly)
library(reshape2)
library(tsne)
library(pls)
library(glmulti)

select <- dplyr::select
source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/performance/cv_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/selection/model_comparisons_lib.R')


##### Data Prep #####

dbu <- get.wide.data(outcome.func=gos.to.binom, scale.vars=F, remove.na.flags=F)
dou <- get.wide.data(outcome.func=gos.to.ord, scale.vars=F, remove.na.flags=F) %>% mutate(gos=factor(gos))
p <- c('age', 'sex', 'marshall', 'gcs')

models <- get.models()
all.vars <- models[['all.vars']]
models <- models[['models']]

results.glm <- run.models(pred.binary.glm, T, ic.score=ic.binary.glm)
results.gbm <- run.models(pred.binary.gbm, T)
results.rf <- run.models(pred.binary.rf, T)

r <- results.glm
cv.res <- get.cv.results(r, 'auc', desc=F)
cv.res <- get.cv.results(r, 'aucpr', desc=F)
cv.res <- get.cv.results(r, 'aicc')
cv.res <- get.cv.results(r, 'aic')

##### ROC Curves #####

model.filter <- c(
  'demo', 'wcov_none', 'wcov_pbto2', 'wcov_icp', 'wcov_icp_pao2', 'wcov_icp_paco2',
  'wcov_icp_pbto2', 'wcov_icp_pao2_pbto2', 'wcov_icp_pao2_pbto2_paco2'
)
plot.roc.curve(r, model.filter, title='GLM ROC')  %>% 
  ggplotly %>% layout(showlegend = T)


##### Individual Models #####

m <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2')) %>% 
  gbm(get.form(models[['wcov_pbto2']]), data = ., n.trees = 1000, distribution='bernoulli',
         interaction.depth = 5, shrinkage=.1, n.minobsinnode = 20)
plot(m, i.var = 5, lwd = 2, col = "blue", main = "")

m <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2')) %>%
  glm(get.form(models[['wcov_icp_pao2_pbto2_paco2']]), data=., family='binomial')
m <- dbu %>% prep.df(c('pbto2')) %>% 
  glm(get.form(models[['wcov_pbto2']]), data=., family='binomial')


##### Exhaustive Modeling #####

d.glmulti <- dbu %>% prep.df(c('icp1', 'pbto2', 'pao2', 'paco2'))
m <- glmulti(get.form(models[['wcov_icp_pao2_pbto2_paco2']]), data=d.glmulti, family='binomial', level=1)
summary(m@objects[[1]])

d.glmulti <- dbu %>% prep.df(c('pbto2', 'pao2'))
m <- glmulti(get.form(models[['wcov_pao2_pbto2']]), data=d.glmulti, family='binomial', level=1)
summary(m@objects[[1]])


##### Predictive Clusters #####

r <- results.glm
cv.data <- r[[1]]$data
cv.diffs <- get.model.diffs(r, 'wcov_icp_pao2_pbto2', 'wcov_icp_pao2')
stopifnot(nrow(cv.data) == nrow(cv.diffs))

cv.llr <- cv.data %>% select(
    one_of(c(p, 'uid', 'gos')), starts_with('icp'), starts_with('pbto2'),
    starts_with('pao2'), starts_with('paco2')
  ) %>% mutate(i=1:nrow(.)) %>% 
  inner_join(cv.diffs %>% select(i, llr), by='i') %>% 
  arrange(desc(llr))


##### Modeling point-wise log-likelihood differences #####
# Try to find significant predictors of LLR

cv.llr %>% select(one_of(p), starts_with('icp'), starts_with('pao2'), llr) %>% 
  # gbm(llr ~ ., data = ., n.trees = 1000, interaction.depth = 5, shrinkage=.1, n.minobsinnode = 20) %>% summary
  # randomForest(llr ~ ., data=.) %>% importance %>% .[order(.[,1]),]
  lm(llr ~ ., data=.) %>% summary
  
cv.llr %>% ggplot(aes(x=pao2_0_300, y=llr)) + geom_point() + stat_smooth(method='lm')
#cv.llr %>% ggplot(aes(x=pao2_0_300, y=age, color=sign(llr))) + geom_point()


library(pls)
X <- cv.llr %>% select(age, marshall, gcs, pao2_0_300, llr)
r <- plsr(llr ~ ., ncomp=4, data=X, validation = "LOO")
#plot(r, plottype = "scores", comps = 1:4)
data.frame(x=r$scores[,1], y=r$scores[,2], llr=X$llr) %>% 
  mutate(llr=cut(llr, breaks=c(-Inf, 0, Inf))) %>% 
  ggplot(aes(x=x, y=y, color=llr)) + geom_point()

#X.mds <- cv.llr %>% select(age, gcs)
X.mds <- tsne(X) %>% apply(2, scale) # dist(X) %>% cmdscale
X.mds %>% as.data.frame %>% setNames(c('x', 'y')) %>%  
  mutate(llr=cv.llr$llr, i=as.numeric(cv.llr$i), pao2_0_300=X$pao2_0_300) %>% 
  mutate(Prediction.Better=ifelse(llr < 0, 'w/o PbtO2', 'w PbtO2')) %>%
  ggplot(aes(x=x, y=y, color=factor(Prediction.Better), size=pao2_0_300)) + 
  geom_jitter(position = position_jitter(width = .1, height = .1)) + 
  theme_bw()



