
library(caret)
library(foreach)
library(iterators)
library(ROCR)

RunLOOCV <- function(data, outcome, covariates, variables){
  #' Computes AUC in LOOCV (note that first level in outcome must be positive class)
  
  # Create an indicator vector for each possible combination of variables
  var.ind <- lapply(seq_along(variables), function(i) c(T, F))
  var.ind <- expand.grid(var.ind)
  
  # Create formulas for all possible models
  form.cov <- paste(outcome, ' ~ ', paste(covariates, collapse=' + '))
  form.all <- apply(var.ind, 1, function(var.mask){
    if (!any(var.mask)) {
      ifelse(length(covariates) == 0, NA, form.cov)
    } else {
      sep <- ifelse(length(covariates) == 0, '', ' + ')
      paste(form.cov, paste(variables[var.mask], collapse=' + '), sep=sep)
    }
  })
  
  # For each model, compute ROC AUC in LOO CV
  outcome.levels <- levels(data[,outcome]) 
  trctrl <- trainControl(method='LOOCV', savePredictions='final', classProbs=T)
  #print(sprintf('Running LOOCV for %s models', length(form.all)))
  foreach(form=form.all, i=icount(), .combine=rbind) %dopar%{
    if (is.na(form)) return(NULL)
    # Calculate AUC using two methods (ROCR and pROC) for the sake of validation
    resamp <- train(as.formula(form), data=data, method='glm', trControl=trctrl)$pred
    pred <- prediction(resamp[,outcome.levels[1]], resamp$obs, label.ordering=rev(outcome.levels))
    auc1 <- performance(pred, 'auc')@y.values[[1]]
    auc2 <- as.numeric(twoClassSummary(resamp, lev=outcome.levels)['ROC'])
    data.frame(formula=form, auc1=auc1, auc2=auc2, stringsAsFactors = F)
  }
}

# Testing
# library(dplyr)
# d <- twoClassSim(n=100)
# X <- d %>% select(-Class)
# y <- d$Class
# 
# X.cov <- X %>% select(starts_with('TwoFactor'), starts_with('Linear')) %>% names
# X.var <- setdiff(names(X), X.cov)
# 
# r <- RunLOOCV(d, 'Class', X.cov, X.var)

