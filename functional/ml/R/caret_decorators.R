#'-----------------------------------------------------------------------------
#' Caret Model Decorators
#'
#' These functions can be used to slightly alter the behavior of caret models
#' to better deal with things like NA predictions.
#' 
#' @author eczech
#'-----------------------------------------------------------------------------


#' @title NA Prediction Handling Decorator
#' @description This decorator will catch NA (or infinite) values in predicted class
#' labels or probabilities for a classifier and attempt to default them to less disruptive values
#' @param model caret model name to decorate; alternatively, a caret modelInfo list
#' @param outcome.levels character vector containing all possible outcome levels for classification
#' problem
#' @param default.class.level index of label in outcome.levels to be used to replace NA values
#' in predicted class labels
#' @return a caret modelInfo object matching \code{model} with \code{prob} and \code{predict}
#' methods overriden to better handle bad values
Decorate.Classifier.NAPredictions <- function(model, outcome.levels, default.class.level=1){
  require(caret)
  
  # Resolve model name to modelInfo list if not already given as modelInfo list
  if (!is.list(model)){
    m <- getModelInfo(model, regex=F)
    if (length(m) > 1)
      stop(sprintf('Found more than one model matching string name "%s"', model))
    m <- m[[1]]
  } else {
    m <- model
    model <- if ('method' %in% names(m)) m$method else 'custom'
  }

  
  old.predict <- m$predict
  new.predict <- function(modelFit, newdata, submodels = NULL){
    p <- old.predict(modelFit, newdata, submodels = submodels)
    
    # If any of the predicted class labels are NA, log a warning and 
    # replace them with some arbitrary default outcome label
    if (any(is.na(p))){
      warning(sprintf('Found NA predicted class labels for %s model', model))
      p <- ifelse(is.na(p), outcome.levels[default.class.level], p)
    }
    p
  }
  m$predict <- new.predict
  
  n.lvl <- length(outcome.levels)
  old.prob <- m$prob
  new.prob <- function(modelFit, newdata, submodels = NULL){
    p <- old.prob(modelFit, newdata, submodels = submodels)
    
    # If any of the predicted probabilities are NA or infinite, log a warning and
    # replace them with equal probabilities per class
    bad.p <- is.na(p) | !is.finite(p)
    if (any(bad.p)){
      warning(sprintf('Found NA predicted probabilities for %s model', model))
      mask <- apply(bad.p, 1, any)
      p[mask, ] <- rep(1/n.lvl, n.lvl)
    }
    p
  }
  m$prob <- new.prob
  
  m
}