library(glmulti)
library(dplyr)


scale <- function(x) (x - mean(x)) / sd(x)
d <- read.csv('/Users/eczech/data/pbto2/export/data_modeling.csv') %>% 
  mutate_each(funs(as.numeric)) %>%
  mutate_each(funs(scale), -gos) %>%
  dplyr::select(-paco2_35_45, -icp1_0_20, -pha_7.35_7.45, -pao2_30_100, -pbto2_20_100, -starts_with('n_'))

d2c <- d %>% mutate(gos = ifelse(gos <= 3, 0, 1))

p <- d2c %>% select(-gos) %>% names
p.main <- c('age', 'marshall', 'gcs', 'sex')
p.sec <- p[!p %in% p.main]

### Glmulti

options(nwarnings=1000)
glm.res.lvl.1 <- glmulti('gos', p, d2c, family='binomial', level=1)

p.int <- c('pbto2_0_20', 'pao2_0_30', 'icp1_20_inf', 'paco2_45_inf')
p.int <- apply(expand.grid(p.int, p.int), 1, function(v) { if (v[1] == v[2]) NA else paste(sort(v), collapse=':')}) 
p.int <- p.int[!is.na(p.int)] %>% unique
form <- as.formula(paste('~ ', paste(c(p, p.int, 'gos'), collapse=' + ')))
mm <- model.matrix(form, d2c)
mm <- mm[,2:ncol(mm)]
glm.data <- data.frame(mm)
glm.res.lvl.2 <- glmulti('gos', c(p, sub(':', '.', p.int)), glm.data, family='binomial', level=1)

#glm.res <- glm.res.lvl.1
glm.res <- glm.res.lvl.2

summary(glm.res)
coef(glm.res)

tmp <- weightable(glm.res)
tmp <- tmp[tmp$aic <= min(tmp$aic) + 2,]
tmp

summary(glm.res@objects[[1]])

# Lasso Selection

library(glmnet)
glmnet.res <- cv.glmnet(as.matrix(d2c[,p]), d2c$gos, family='binomial')
coefs <- glmnet.res$glmnet.fit %>% coef
selection.order <- coefs %>% as.matrix %>% apply(1, function(x){ 
  i = 1:length(x)
  i[abs(x) > 0][1]
})
sort(selection.order)

library(randomForest)
rf.res <- randomForest(factor(gos) ~ ., data=d2c)
rf.res$importance %>% data.frame %>% add_rownames %>% arrange(MeanDecreaseGini)

library(Boruta)
b.res <- Boruta(d2c[,p], factor(d2c$gos))
attStats(b.res) %>% add_rownames %>% arrange(meanZ)
