#'-----------------------------------------------------------------------------
#' ML Training Template (Manual Version)
#'
#' This module contains code for running a set of ML models (with no complicated
#' feature selection in resampling), caching those models on disk, and plotting
#' their respective performance as well as partial dependence.
#' 
#' Note that this template does not use the MLProject abstraction but rather
#' uses file system management directly.
#' 
#' @seealso ./project_training.R
#' @author eczech
#'-----------------------------------------------------------------------------

##### Initialization #####

# Base Imports
library(plyr); library(dplyr)
source('~/repos/portfolio/functional/common/R/utils.R')

options(import.root='~/repos')
import_source('portfolio/functional/common/R/cache.R')
import_source('portfolio/functional/ml/R/trainer.R')
import_source('portfolio/functional/ml/R/results.R')
import_source('portfolio/functional/ml/R/parallel.R')
import_source('portfolio/functional/ml/R/partial_dependence.R')

# Imports for modeling project
library(caret)

# Pick a seed
SEED <- 123
  
# Choose a location for raw and cache data
CACHE_DIR <- '~/data/unit_tests/basic_trainer/cache'
DATA_DIR <- '~/data/unit_tests/basic_trainer/data'

##### Load Data #####

d <- read.csv(file.path(DATA_DIR, 'sim.csv'))
X <- d %>% select(-Class)
y <- d$Class


##### Model Training #####

# Initialize training parameters
tr <- SimpleTrainer(cache.dir=CACHE_DIR, cache.project='models', seed=SEED)
tc <- trainControl(
  method='cv', number=10, classProbs=T, 
  summaryFunction = function(...)c(twoClassSummary(...), defaultSummary(...)),
  verboseIter=T, savePredictions='final', returnResamp='final'
)

# Register multicore backend
registerCores(1)

# Define models to train over
models <- list(
  tr$getModel('glm', method='glm', preProcess=c('center', 'scale'), trControl=tc),
  tr$getModel('rpart', method='rpart', tuneLength=10, trControl=tc)
)
names(models) <- sapply(models, function(m) m$name)

# Fit resampled models
results <- lapply(models, function(m) tr$train(m, X, y, enable.cache=T)) %>% setNames(names(models))


##### Performance Evaluation #####

# Plot performance measures using directly attached resampled statistics
resamp <- GetResampleData(results)
GetDefaultPerfPlot(resamp, 'accuracy')

# Plot performance measures computed on-the-fly (should be identical results in this case)
perf <- GetPerfData(results)
GetDefaultPerfPlot(resamp, 'roc')


##### Inference #####

# Variable Importance
var.imp <- GetVarImp(results)
PlotVarImp(var.imp, compress=F)

### Partial Dependence

pd.vars <- c('Linear09', 'TwoFactor1') # List of variables to get PD for
pd.models <- c('glm', 'rpart')         # List of model names to get PD for

pred.fun <- function(object, newdata) {
  require(caret)
  pred <- predict(object, newdata=newdata, type='prob')
  if (is.vector(pred)) pred
  else pred[,1] # Make sure this is the right predicted probability for your problem
}

registerCores(1) # Increase this to make PD calcs faster
pd.data <- GetPartialDependence(
  results[pd.models], pd.vars, pred.fun, 
  X=X, # This can come from model objects but only if returnData=T in trainControl
  grid.size=50, grid.window=c(0, 1), # Resize these to better fit range of data
  sample.rate=1, # Decrease this if PD calculations take too long
  verbose=T, seed=SEED
)
PlotPartialDependence(pd.data, se=F, mid=T, use.smooth=T, facet.scales='free_x')


