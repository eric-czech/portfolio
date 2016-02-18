
library(caret)
library(dplyr)
library(stringr)
library(foreach)

set.seed(123)
d <- twoClassSim(n=500, noiseVars=2000, linearVars = 10, )
scale <- function(x) (x - mean(x)) / sd(x)
X <- d %>% select(-Class) %>% mutate_each(funs(scale))
y <- d$Class

mfit <- rda::rda(t(as.matrix(X)), as.integer(y), alpha=.9999, delta=10, genelist = T)
mm <- rda::rda(t(as.matrix(X)), as.integer(y), genelist = T)
mcv <- rda::rda.cv(mm, t(as.matrix(X)), as.integer(y))
plot(mcv)
predict.rda(m, x=t(as.matrix(X)), y=as.integer(y), xnew=t(as.matrix(X)), alpha=0, delta=.1, type='class')

predict.rda(mfit, x=t(as.matrix(X)), y=as.integer(y), xnew=NULL, type='nonzero')

GetPLSModel <- function(){
  m <- getModelInfo('pls', regex=F)[[1]]
  m$loop <- function (grid) {
    grid <- grid[order(grid$ncomp, decreasing = TRUE), , drop = FALSE]
    loop <- grid[1, , drop = FALSE]
    submodels <- list(grid[-1, , drop = FALSE])
    list(loop = loop, submodels = submodels)
  }
  m$fit <- function (x, y, wts, param, lev, last, classProbs, ...) {
    out <- if (is.factor(y)) {
      plsda(x, y, method = "oscorespls", ncomp = param$ncomp,
            ...)
    }
    else {
      dat <- if (is.data.frame(x))
        x
      else as.data.frame(x)
      dat$.outcome <- y
      plsr(.outcome ~ ., data = dat, method = "oscorespls",
           ncomp = param$ncomp, ...)
    }
    out
  }
  m$varImp <- function (object, estimate = NULL, ...) {
    modelCoef <- coef(object, intercept = FALSE, comps = 1:object$ncomp)
    perf <- MSEP(object)$val
    nms <- dimnames(perf)
    if (length(nms$estimate) > 1) {
      pIndex <- if (is.null(estimate))
        1
      else which(nms$estimate == estimate)
      perf <- perf[pIndex, , , drop = FALSE]
    }
    numResp <- dim(modelCoef)[2]
    if (numResp <= 2) {
      modelCoef <- modelCoef[, 1, , drop = FALSE]
      perf <- perf[, 1, ]
      delta <- -diff(perf)
      delta <- delta/sum(delta)
      out <- data.frame(Overall = apply(abs(modelCoef), 1,
                                        weighted.mean, w = delta))
    }
    else {
      perf <- -t(apply(perf[1, , ], 1, diff))
      perf <- t(apply(perf, 1, function(u) u/sum(u)))
      out <- matrix(NA, ncol = numResp, nrow = dim(modelCoef)[1])
      for (i in 1:numResp) {
        tmp <- abs(modelCoef[, i, , drop = FALSE])
        out[, i] <- apply(tmp, 1, weighted.mean, w = perf[i,
                                                          ])
      }
      colnames(out) <- dimnames(modelCoef)[[2]]
      rownames(out) <- dimnames(modelCoef)[[1]]
    }
    browser()
    as.data.frame(out)
  }
  m
}

pls.fit <- train(X, y, method=GetPLSModel(), tuneGrid=data.frame(ncomp=1:5), trControl=trainControl(classProbs=T))
varImp(pls.fit)

GetSCRDAModel <- function(max.delta=3){
  m <- list(type = "Classification", library = "rda", loop = NULL)
  params <- data.frame(
    parameter=c('alpha', 'delta', 'len'),
    class=c('numeric', 'numeric', 'numeric'),
    label=c('#Alpha', '#Delta', '#Params')
  )
  get.predictions <- function(modelFit, newdata, submodels, type, transform){
    m <- modelFit
    x <- newdata

    if (!is.matrix(x))
      x <- as.matrix(x)
    x <- t(x)

    params <- rbind(modelFit$loop.params, submodels)
    alpha <- sort(unique(params$alpha))
    delta <- sort(unique(params$delta))
    pred <- rda::predict.rda(m, m$x.train, m$y.train, x, alpha=alpha, delta=delta, type=type)

    if (is.null(submodels))
      return(transform(pred))

    stopifnot(length(alpha) == dim(pred)[1])
    stopifnot(length(delta) == dim(pred)[2])

    pred <- foreach(i=1:length(alpha), .combine=c)%:%foreach(j=1:length(delta))%do%{
      if (type == 'posterior') transform(pred[i, j, , ])
      else transform(pred[i, j, ])
    }
    pred
  }
  m <- list(
    label = "SCRDA",
    library = c("rda"),
    type = "Classification",
    parameters = params,
    grid = function(x, y, len = NULL, search = "grid") {
      grid <- expand.grid(alpha=seq(0, 0.99, len=len), delta=seq(0, max.delta, len=len))
      cbind(grid, data.frame(len=len))
    },
    sort = function(x) {
      x[order(x$alpha, x$delta),]
    },
    loop = function (grid) {
      grid <- grid[order(grid$alpha, grid$delta), , drop = F]
      loop <- grid[1, , drop = F]
      submodels <- list(grid[-1, , drop = F])
      list(loop = loop, submodels = submodels)
    },
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
      if (!is.factor(y))
        stop('Response must be a factor for SCRDA models')
      print('Fit')
      if (is.data.frame(x))
        x.names <- names(x)
      else
        x.names <- NULL
      if (!is.matrix(x))
        x <- as.matrix(x)
      x <- t(x)
      y <- as.integer(y)

      alpha <- seq(0, 0.99, len=param$len)
      delta <- seq(0, max.delta, len=param$len)
      m <- rda::rda(x, y, alpha=alpha, delta=delta, genelist = T)
      m$x.names <- x.names
      m$x.train <- x
      m$y.train <- y
      m$loop.params <- param
      m
    },
    predict = function(modelFit, newdata, submodels = NULL) {
      print('Prediction')
      transform <- function(pred) {
        sapply(as.integer(pred), function(p) {
          if (p == 1) modelFit$obsLevel[1]
          else if (p == 2) modelFit$obsLevel[2]
          else NA
        })
      }
      get.predictions(modelFit, newdata, submodels, 'class', transform)
    },
    prob = function(modelFit, newdata, submodels = NULL) {
      print('Prob')
      transform <- function(pred) {
        dimnames(pred) <- unname(dimnames(pred))
        dimnames(pred)[[2]] <- modelFit$obsLevels
        pred
      }
      get.predictions(modelFit, newdata, submodels, 'posterior', transform)
    },
    varImp = function(object, estimate = NULL, ...) {
      params <- list(...)
      if (is.null(params$alpha) || is.null(params$delta))
        stop('Variable importance for SCRDA is only possible when specifying alpha and delta arguments')

      var.imp <- predict.rda(object, x=object$x.train, y=object$y.train,
                             xnew=NULL, alpha=params$alpha, delta=params$delta, type='nonzero')
      var.imp <- data.frame(Overall=var.imp)
      if (!is.null(object$x.names))
        rownames(var.imp) <- object$x.names
      var.imp
    },
    predictors = function(x, ...) {
      browser()
      rownames(x$projection)
    },
    levels = function(x) {
      browser()
      x$obsLevels
    }
  )
}

set.seed(123)
library(doMC)
library(glmnet)
registerDoMC(1)

set.seed(123)
fit <- train(X, y, method=GetSCRDAModel(3), tuneLength=4,
             trControl=trainControl(method='cv', number=2,
                                    classProbs=T, savePredictions = T))
predict(fit, X)


init <- glmnet(as.matrix(X), y, family = 'binomial', nlambda = 100, alpha = 1)
lambda <- unique(init$lambda)
lambda <- lambda[-c(1, length(lambda))]
lambda <- lambda[1:length(lambda)]
ref.fit <- train(X, y, method='glmnet',
                 tuneGrid=data.frame(alpha=1, lambda=lambda),
                 trControl=trainControl(method='cv', number=5, classProbs=T))

var.imp <- varImp(ref.fit)
var.imp <- var.imp$importance
var.imp$Overall <- ifelse(var.imp$Overall > 0, 1, 0)

var.imp <- varImp(fit, alpha=fit$bestTune$alpha, delta=fit$bestTune$delta, scale=F)
var.imp <- var.imp$importance
#var.imp <- varImp(pls.fit, scale=F)

var.imp %>% add_rownames(var = 'feature') %>%
  mutate(is.noise=str_detect(feature, 'Noise')) %>%
  rename(value=Overall) %>% mutate(type=str_replace_all(feature, '\\d+', '')) %>%
  group_by(value, type) %>% tally

var.imp$importance[order(var.imp$importance),,drop=F]
