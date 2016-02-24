
#' This is a test script for the "High dimensional regularized discriminant
#' classifier" model outlined here:
#' 
#' https://github.com/ramhiser/sparsediscrim/blob/master/R/smdqda.r
#' 
#' The original paper for this model is here: 
#' 
#' http://arxiv.org/pdf/1602.01182v1.pdf
#' 
#' Note that the source for the above is not in sync with CRAN.  See
#' here for actual source: https://github.com/cran/sparsediscrim
#' 

library(sparsediscrim)

library(caret)
library(dplyr)

d <- twoClassSim(n=100, noiseVars = 100)
scale <- function(x)(x-mean(x))/sd(x)
X <- d %>% select(-Class) %>% mutate_each(funs(scale))
y <- d$Class

# Test basic functionality
m <- hdrda(X, y)
p <- predict(m, X)


# Test caret model

GetHDRDAModel <- function(){
  list(label = "High Dimensional Regularized Discriminant Analysis",
       library = "sparsediscrim",
       loop = NULL,
       type = c('Classification'),
       parameters = data.frame(parameter = c('lambda', 'gamma', 'shrinkage'),
                               class = c('numeric', 'numeric', 'character'),
                               label = c('Lambda', 'Gamma', 'Shrinkage Type')),
       grid = function(x, y, len = NULL, search = "grid") {
         # See recommended hyperparameter settings at:
         # https://github.com/ramhiser/sparsediscrim/blob/master/R/hdrda.r#L315
         if(search == "grid") {
           lambda <- seq(0, 1, len = len)
           gamma_ridge <- c(0, 10^seq.int(-2, 4, len = len-1))
           gamma_convex <- seq(0, 1, len = len)
         } else {
           lambda <- runif(len, min = 0, max = 1)
           gamma_ridge <- runif(len, min = 10^-2, max = 10^4)
           gamma_convex <- runif(len, min = 0, max = 1)
         }
         out <- rbind(
           expand.grid(lambda=lambda, gamma=gamma_ridge, shrinkage='ridge'),
           expand.grid(lambda=lambda, gamma=gamma_convex, shrinkage='convex')
         )
         out$shrinkage <- as.character(out$shrinkage)
         out
       },
       fit = function(x, y, wts, param, lev, last, classProbs, ...) {
         sparsediscrim::hdrda(x, y, lambda=param$lambda, gamma=param$gamma, shrinkage_type=param$shrinkage)
       },
       predict = function(modelFit, newdata, submodels = NULL) {
         r <- predict(modelFit, newdata)
         r$class
       },
       prob = function(modelFit, newdata, submodels = NULL) {
         r <- predict(modelFit, newdata)
         r$posterior
       },
       levels = function(x) x$obsLevels,
       tags = c("Discriminant Analysis", "Linear Classifier"),
       sort = function(x) x[order(x$lambda, x$gamma, x$shrinkage),])
}

m <- train(X, y, method=GetHDRDAModel(), tuneLength=5, 
           trControl=trainControl(classProbs=T, savePredictions='final'))

grid <- expand.grid(lambda=seq(0, 1, len=3), gamma=c(.0001, .001, .01, .1, 0), shrinkage='ridge', stringsAsFactors = F)
m <- train(X, y, method=GetHDRDAModel(), tuneGrid=grid,
           trControl=trainControl(classProbs=T, savePredictions='final'))

