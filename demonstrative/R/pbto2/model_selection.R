library(glmulti)
library(dplyr)
library(MASS)

select <- dplyr::select
source('/Users/eczech/repos/portfolio/demonstrative/R/pbto2/common.R')

#do <- get.wide.data(outcome.func=gos.to.ord)
db <- get.wide.data(outcome.func=gos.to.binom)

p <- db %>% select(-gos) %>% names
p.main <- c('age', 'marshall', 'gcs', 'sex')
p.sec <- p[!p %in% p.main]


### Glm
glm.res.1 <- db %>% glm(gos ~ ., data=., family='binomial')

### Glmulti

glm.res.lvl.1 <- glmulti('gos', p, db, family='binomial', level=1)


# glmulti with interactions
p.int <- c('pbto2_0_20', 'pao2_0_30', 'icp1_20_inf', 'paco2_45_inf')
p.int <- apply(expand.grid(p.int, p.int), 1, function(v) { if (v[1] == v[2]) NA else paste(sort(v), collapse=':')}) 
p.int <- p.int[!is.na(p.int)] %>% unique
form <- as.formula(paste('~ ', paste(c(p, p.int, 'gos'), collapse=' + ')))
mm <- model.matrix(form, db)
mm <- mm[,2:ncol(mm)]
glm.data <- data.frame(mm)
glm.res.lvl.2 <- glmulti('gos', c(p, sub(':', '.', p.int)), glm.data, family='binomial', level=1)

glm.res <- glm.res.lvl.1
#glm.res <- glm.res.lvl.2

save(glm.res, file='/Users/eczech/data/pbto2/export/glmulti_res_no_interp.Rdata')

# glmulti.env <- new.env()
# glmulti.res <- load(file='/Users/eczech/data/pbto2/export/glmulti_res_all_interp.Rdata', envir=glmulti.env)
# glm.res.old <- glmulti.env$glm.res

summary(glm.res)
coef(glm.res)

tmp <- weightable(glm.res)
tmp <- tmp[tmp$aic <= min(tmp$aic) + 2,]
tmp

summary(glm.res@objects[[1]])

# Lasso Selection

library(glmnet)
glmnet.res <- cv.glmnet(as.matrix(db[,p]), db$gos, family='binomial')
coefs <- glmnet.res$glmnet.fit %>% coef
selection.order <- coefs %>% as.matrix %>% apply(1, function(x){ 
  i = 1:length(x)
  i[abs(x) > 0][1]
})
sort(selection.order)

library(randomForest)
rf.res <- randomForest(factor(gos) ~ ., data=db)
rf.res$importance %>% data.frame %>% add_rownames %>% arrange(MeanDecreaseGini)

library(Boruta)
b.res <- Boruta(db[,p], factor(db$gos))
attStats(b.res) %>% add_rownames %>% arrange(meanZ)
