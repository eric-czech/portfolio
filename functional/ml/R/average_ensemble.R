aggregating_model <- list(
  label = "Averaging Model",
  library = c(),
  type = "Classification",
  parameters = data.frame(parameter='aggfunc', class='character', label='Aggregation Function'),
  grid = function(x, y, len = NULL, search = "grid") data.frame(aggfunc = c('mean')),
  loop = NULL,
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    list(y=y, lev=lev)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    browser()
    p <- apply(newdata, 1, mean)
    p <- ifelse(p < .5, modelFit$lev[1], modelFit$lev[2])
    factor(p, levels=modelFit$lev)
  },
  prob = function(modelFit, newdata, submodels = NULL){
    p <- apply(newdata, 1, mean)
    p <- cbind(1-p, p)
    dimnames(p)[[2]] <- modelFit$obsLevels
    p
  },
  varImp = NULL,
  predictors = function(x, ...) predictors(x$terms),
  levels = function(x) if(any(names(x) == "obsLevels")) x$obsLevels else NULL,
  sort = NULL
)

library(caret)
library(dplyr)
d <- twoClassSim()
X <- d %>% select(-Class)
y <- d$Class
m <- train(X, y, method=aggregating_model, 
      trControl=trainControl(method='cv', number=10, classProbs=T))


