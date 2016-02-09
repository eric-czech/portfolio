library(foreach)
library(dplyr)

SummarizeResults = function(results, fold.summary=NULL, model.summary=NULL){
  
  preds <- foreach(m=results, .combine=rbind) %do% {
    foreach(fold=m, .combine=rbind) %do% {
      data.frame(
        fold=fold$fold, y.pred=fold$y.pred, 
        y.test=fold$y.test, model=fold$model
      )
    }
  }
  
  # Summarize per-fold results, if applicable
  if (!is.null(fold.summary)) fold.summary <- preds %>% group_by(model, fold) %>% do({fold.summary(.)})
  else fold.summary <- NULL
  
  # Summarize per-model results, if applicable
  if (!is.null(model.summary)) model.summary <- preds %>% group_by(model) %>% do({model.summary(.)})
  else model.summary <- NULL
  
  list(model.summary=model.summary, fold.summary=fold.summary)
}
