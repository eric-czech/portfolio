library(foreach)
library(dplyr)

#####################################################
##### Trainer specific result summary utilities #####
#####################################################

#' @title Result summary utilities for models built using \code{trainer}
SummarizeTrainingResults = function(results, is.classification, fold.summary=NULL, model.summary=NULL){
  
  preds <- foreach(m=results, .combine=rbind) %do% {
    foreach(fold=m, .combine=rbind) %do% {
      res <- data.frame(fold=fold$fold, y.test=fold$y.test, model=fold$model)
      if (is.classification){
        res$y.pred.class <- fold$y.pred$class
        res$y.pred.prob <- fold$y.pred$prob
      } else {
        res$y.pred <- fold$y.pred
      }
      res
    }
  }
  
  # Summarize per-fold results, if applicable
  if (!is.null(fold.summary)) fold.summary <- preds %>% group_by(model, fold) %>% do({fold.summary(.)})
  else fold.summary <- NULL
  
  # Summarize per-model results, if applicable
  if (!is.null(model.summary)) model.summary <- preds %>% group_by(model) %>% do({model.summary(.)})
  else model.summary <- NULL
  
  list(predictions=preds, model.summary=model.summary, fold.summary=fold.summary)
}


############################################
##### Generic result summary utilities #####
############################################

##### Resampling Summarizations #####

#' @title Default metric extractor for use with \code{GetPerfData}
#' @param pred resampled prediction data from a caret::train object
#' @return a data frame containing CM cell counts as well as 
#' roc, accuracy, kappa, etc.
DefaultMetricExtractor <- function(pred){
  require(caret)
  require(dplyr)
  
  cmat <- caret::confusionMatrix(pred$pred, pred$obs)
  roc <- twoClassSummary(data.frame(pred), lev=levels(pred$obs))
  
  data.frame(
    tp=cmat$table[1,1],
    fp=cmat$table[1,2],
    fn=cmat$table[2,1],
    tn=cmat$table[2,2],
    roc=as.numeric(roc[['ROC']]),
    acc=as.numeric(cmat$overall[['Accuracy']]),
    kappa=as.numeric(cmat$overall[['Kappa']]),
    spec=as.numeric(cmat$byClass['Sensitivity']),
    sens=as.numeric(cmat$byClass['Specificity'])
  )
}

#' @title Extracts performance and CM metrics for several trained models
#' @param models a list of caret::train objects 
#' @note The given train objects must have had savePredictions='final' in 
#' trainControl (results will be misleading if savePredictions='all')
#' @return A data frame containing performance measures for each model
GetPerfData <- function(models, metric.extraction.fun=DefaultMetricExtractor){
  require(foreach)
  require(dplyr)
  
  if (is.null(metric.extraction.fun))
    stop('Metric extraction function must be specified')
  
  foreach(m=names(models), .combine=rbind)%do%{
    model <- models[[m]]
    
    if (!any(c('train', 'caretStack', 'caretEnsemble') %in% class(model))){
      warning(sprintf('Ignoring model with incompatible class "%s"', class(model)))
      return(NULL)
    }
    
    # Verify existence of resampling predictions
    has.train.model <- !is.null(model$pred)
    has.train.ens   <- !is.null(model$ens_model) && !is.null(model$ens_model$pred)
    if (!has.train.model && !has.train.ens)
      stop('Model must have resampled prediction data ("final" ONLY)')
    
    pred <- if (has.train.model) model$pred else model$ens_model$pred
    has.dupes <- pred %>% group_by(Resample, rowIndex) %>% tally %>% .$n %>% max
    if (has.dupes > 1)
      stop(paste(
        'Found repeated observations in resample predictions. ',
        'Saved predictions should be "final" ONLY'
      ))
    
    pred %>% dplyr::rename(resample=Resample) %>% 
      group_by(resample) %>% do({ metric.extraction.fun(.) }) %>%
      dplyr::mutate(model=m)
  }
}

#' @title Plots performance measures extracted using \code{DefaultMetricExtractor}
#' @param perf.data data frame resulting from call to \code{GetPerfData}
#' @param metric name of metric extracted to plot for each model
#' @note The given object \code{perf.data} should come from \code{GetPerfData} but
#' if that function was called with a non-default metric extractor, this plotting
#' function will not work (presumably it was overriden with the intention of 
#' plotting in some alternative way)
GetDefaultPerfPlot <- function(perf.data, metric, metric.title=NULL){
  require(dplyr)
  require(ggplot2)
  
  perf.data %>% 
    dplyr::rename_(value=metric) %>%
    dplyr::mutate(model=factor(model)) %>%
    dplyr::mutate(model=reorder(model, value)) %>%
    ggplot(aes(x=model, y=value, color=model)) + 
    geom_boxplot(outlier.size=0) + 
    geom_jitter(alpha=.2, width=.5) + theme_bw() + 
    ylab(ifelse(is.null(metric.title), metric, metric.title)) +
    scale_colour_discrete(guide = FALSE) + xlab('Model') +
    ggtitle('Model Performance')
}

#' @title Plots confusion matrix counts extracted using \code{DefaultMetricExtractor}
#' @param perf.data data frame resulting from call to \code{GetPerfData}
#' @note The given object \code{perf.data} should come from \code{GetPerfData} but
#' if that function was called with a non-default metric extractor, this plotting
#' function will not work (presumably it was overriden with the intention of 
#' plotting in some alternative way)
GetDefaultCMPlot <- function(perf.data){
  require(dplyr)
  require(ggplot2)
  require(reshape2)
  
  cm <- c('tp', 'fp', 'fn', 'tn')
  cm.label <- c('True Positive', 'False Positive', 'False Negative', 'True Negative')
  perf.data %>% 
    melt(id.vars=c('resample', 'model'), measure.vars=cm) %>%
    dplyr::mutate(variable=factor(as.character(variable), levels=cm, labels=cm.label)) %>%
    ggplot(aes(x=model, y=value, color=model)) + 
    geom_boxplot() + geom_jitter(alpha=.3) +
    facet_wrap(~variable, scales='free') + theme_bw()
}

##### Variable Importance #####

#' @title Aggregate feature importance across many models
#' @param models list of caret::train objects (w/ varImp)
#' @return data frame containing variable importances for each model
GetVarImp <- function(models){
  require(dplyr)
  require(caret)
  require(foreach)
  
  foreach(m=names(models), .combine=rbind) %do% {
    model <- models[[m]]
    
    is.ens <- 'caretEnsemble' %in% class(model)
    
    # Ignore caret stacked ensembles, they do not have var imp
    if (!is.ens && 'caretStack' %in% class(model))
      return(NULL)
    
    if (!is.ens && !'train' %in% class(model)){
      warning(sprintf('Ignoring model with incompatible class "%s"', class(model)))
      return(NULL)
    }

    vimp <- varImp(model)
    if (is.null(vimp)) return(NULL)
    
    if (is.ens) vimp <- vimp[,'overall',drop=F]
    else {
      vimp <- vimp$importance
      # Ignore when multiple measures of variable importance are returned
      # for non-ensemble model (not sure what to do with those yet)
      if (ncol(vimp) > 1) return(NULL)
    }

    setNames(vimp, 'score') %>%
      add_rownames(var='feature') %>% 
      dplyr::mutate(model=m)
  }  
}

#' @title Plot results from \code{GetVarImp}
PlotVarImp <- function(var.imp, limit=15, compress=F){
  require(dplyr)
  require(ggplot2)
  
  var.imp <- var.imp %>% 
    dplyr::mutate(feature=reorder(factor(feature), score, FUN=mean, order=T)) %>%
    dplyr::filter(feature %in% tail(levels(feature), limit))
  
  if (compress){
    ggplot(var.imp, aes(x=feature, y=score)) + 
      geom_boxplot(outlier.size=0) +
      geom_jitter(aes(x=feature, y=score, color=model), width=.5, alpha=.5) + 
      theme_bw() + ggtitle('Feature Importance Across Models') +
      theme(axis.text.x = element_text(angle = 25, hjust = 1)) 
  } else {
    ggplot(var.imp, aes(x=feature, y=score, color=model)) + geom_point() +
      theme_bw() + ggtitle('Feature Importance by Model') +
      theme(axis.text.x = element_text(angle = 25, hjust = 1))
  }

}

##### Partial Dependence #####

GetPartialDependence <- function(
  models, vars, pred.fun, X=NULL, grid.size=50, grid.window=c(0, 1), 
  sample.rate=1, verbose=verbose, seed=NULL){
  
  require(dplyr)
  require(reshape2)
  
  if (!exists("partialDependence")){
    stop(paste0(
      'Function "partialDependence" not defined.  ',
      'Try "source_url(\'http://cdn.rawgit.com/eric-czech/portfolio/master/functional/ml/R/trainer.R\') " ',
      'and run again.'
    ))
  }
  
  get.training.data <- function(model){
    if (!is.null(model$trainingData)) 
      return(model$trainingData)
    if (!is.null(model$models) && !is.null(model$models[[1]]$trainingData))
      return(model$models[[1]]$trainingData)
    return(NULL)
  }
  
  use.training.data <- F
  if (is.null(X)){
    mask <- sapply(models, function(x) is.null(get.training.data(x)))
    bad.models <- names(models)[mask]
    if (length(bad.models) > 0){
      stop(paste0(
        'If no covariate data is given for partial dependence calculations ',
        'then it is expected that the trainingData attribute is retained in ',
        'model training.  Rerun after retraining models with "savePredictions=\'final\'" ',
        'in trainControl or pass the predictor data explicitly using the "X" argument.  The ',
        'following models were found to have no training data: ', paste(bad.models, collapse=', ')
      ))
    }
    use.training.data <- T
  }

  pd.fun <- function(m, var){
    if (verbose)
      cat(sprintf('Calculating "%s" partial dependence for model "%s"\n', var, m))
    
    # Compute partial dependence for single variable + model
    model <- models[[m]]
    
    if (use.training.data)
      X <- get.training.data(model) %>% dplyr::select(-.outcome)
    
    if (!is.null(seed))
      set.seed(seed)
    
    pd <- partialDependence(
      model, X, var, pred.fun, 
      grid.size=grid.size, grid.window=grid.window, 
      sample.rate=sample.rate, verbose=verbose
    )
    
    # Convert wide matrix for PD to long format (easier to plot that way)
    pd.df <- pd$pdp %>% as.data.frame %>% add_rownames(var='x')
    if (!is.factor(pd$x))
      pd.df <- pd.df %>% mutate(x=as.numeric(x))
    pd.df <- pd.df %>% 
      melt(id.vars='x', value.name = 'y', variable.name='i')
    
    list(pd=pd.df, x=pd$x, predictor=var, model=m)
  }
  arg <- expand.grid(names(models), vars, stringsAsFactors = F)
  mapply(pd.fun, arg[, 1], arg[, 2], SIMPLIFY = F, USE.NAMES = F)
}

PlotPartialDependence <- function(pd.data){
  is.discrete <- sapply(pd.data, function(r) is.factor(r$x))
  pd.discrete.data <- foreach(pd=pd.data[is.discrete], .combine=rbind)%do%{
    pd$pd %>% mutate(predictor=pd$predictor, model=pd$model)
  }
  pd.numeric.data <- foreach(pd=pd.data[!is.discrete], .combine=rbind)%do%{
    pd$pd %>% mutate(predictor=pd$predictor, model=pd$model)
  }
  
  pd.discrete <- NULL
  if (is.data.frame(pd.discrete.data) && nrow(pd.discrete.data) > 0){
    pd.discrete <- ggplot(pd.discrete.data, aes(x=x, y=y, color=model)) + 
      geom_boxplot(position='dodge') + 
      facet_wrap(~predictor) + theme_bw()
  }
  
  pd.numeric <- NULL
  if (is.data.frame(pd.numeric.data) && nrow(pd.numeric.data) > 0){
    pd.numeric <- ggplot(pd.numeric.data, aes(x=x, y=y, color=model)) +
      geom_smooth(size=1) + facet_wrap(~predictor) + theme_bw()
  }
  
  list(pd.numeric=pd.numeric, pd.discrete=pd.discrete)
}
