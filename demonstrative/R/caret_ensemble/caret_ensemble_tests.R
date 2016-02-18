# This is a stand-alone example used to replicated a bug discussed here:
# https://github.com/zachmayer/caretEnsemble/pull/190

library(MASS); library(plyr); library(dplyr)
library(caret)
library(devtools)
# unload(inst("caretEnsemble")); install_local('/Users/eczech/repos/misc/caretEnsemble'); library(caretEnsemble)
library(caretEnsemble)


### Single Example ###
set.seed(SEED)
X <- matrix(c(rnorm(n = 1500, mean=5), rnorm(n = 1500, mean=-5)), nrow = 1000, ncol=3)
p <- 1 / (1 + exp(-apply(X, 1, function(x) sum(x))))
cl <- ifelse(sapply(p, function(x) runif(1) < x), 'yes', 'no')
y <- factor(cl, levels=c('no', 'yes'))

train.idx <- createDataPartition(y)[[1]]
glm.model <- train(X[train.idx,], y[train.idx], method='glm')
glm.pred <- predict(glm.model, newdata=X[-train.idx,], type='raw')
glm.prob <- predict(glm.model, newdata=X[-train.idx,], type='prob')
confusionMatrix(table(glm.pred, y[-train.idx]), positive = 'yes')

model.list <- caretList(
  X[train.idx,], y[train.idx],
  trControl=trainControl(method="cv", number=10, savePredictions="final", classProbs=T),
  methodList=c("glmnet", "rda", "rf")
)
ens.model <- caretEnsemble(model.list)
ens.pred <- predict(ens.model, newdata=X[-train.idx,], type='raw')
confusionMatrix(table(ens.pred, y[-train.idx]), positive='yes')

