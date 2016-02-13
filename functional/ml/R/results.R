library(foreach)
library(dplyr)

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
