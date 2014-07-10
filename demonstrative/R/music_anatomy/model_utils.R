# Utilities for model cross validation

library(cvTools)
library(DMwR)
library(logging)

model_utils = new.env()

model_utils$cross_validate = function(data, trainer, predictor, response_var, k = 10, .type='random', .fetch_var_imp = NULL, .fetch_classes = NULL, .smote_training=F){
  #' Runs cross-validation on the given data set using the given 'trainer' 
  #' function to build a model for a fold and then uses the 'predictor' to
  #' produce predictions on a validation dataset for that same fold.
  #'
  #' The results returned will include, per fold number:
  #' 1) Training and validation datasets
  #' 2) Model fit
  #' 3) Predicted response probabilities
  #' 4) Actual response value
  #' And optionally:
  #' 5) Predicted response classes
  #' 6) Variable importance
  #' 
  #' Args:
  #'   data: full data set to be modeled 
  #'   trainer: function used to fit a model; should accept a subset of the given
  #'     data (with the same form) and return some fit model object
  #'   predictor: function used to predict response values given a model fit and
  #'     validation data to produce predictions for (result should be a single vector of predicted values)
  #'   response_var: name of response variable
  #'   k: number of folds to use
  #'   .type: type of folding to use; defaults to 'random' but 'interleaved' or 'consecutive' can be used for deterministic folds
  #'   .fetch_var_imp: function used to fetch variable importance within model fit [Optional]
  #'   .fetch_classes: function used to fetch predicted classes (not probabilities) using model fit [Optional]
  #'   .smote_training: flag indicating whether or not training sets should be passed through SMOTE 
  #'      (http://cran.r-project.org/web/packages/DMwR/DMwR.pdf#page=82) for downsampling
  folds = cvFolds(n=nrow(data), K=k, type=.type)
  results = list()
  for (i in 1:k){    
    loginfo(paste0('cv> Running model for fold ', i,' ...'))
    results$training[[i]] = data[folds$subsets[folds$which != i],]
    if (.smote_training){
      results$training[[i]] = SMOTE(as.formula(paste0(response_var, '~.')), results$training[[i]])
    }
    results$validation[[i]]  = data[folds$subsets[folds$which == i],]
    results$fit[[i]]         = trainer(results$training[[i]])
    results$predictions[[i]] = predictor(results$fit[[i]], results$validation[[i]])
    results$response[[i]]    = results$validation[[i]][,response_var]
    if (!is.null(.fetch_var_imp)){
      results$var_imp[[i]] = .fetch_var_imp(results$fit[[i]])
    }
    if (!is.null(.fetch_classes)){
      results$classes[[i]] = .fetch_classes(results$fit[[i]], results$validation[[i]])
    }
  }
  results
}
