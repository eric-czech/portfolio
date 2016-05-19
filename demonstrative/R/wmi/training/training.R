#'-----------------------------------------------------------------------------
#' ML Project Training Template
#'
#' This module contains skeleton code for training ML models within the context
#' of some greater project.
#' 
#' @author eczech
#'-----------------------------------------------------------------------------
source('common.R')


##### Load Data #####

d <- proj$getData('raw.csv')

meta.cols <- c('AssessmentID', 'AssessmentName', 'CommissioningDate', 'InstallationDate')
idx.cols <- c(
  'Feasible', 'Priority', 'ProjectClassification', 'ProjectManager', 'ProjectSubclassification', 
  'ProjectType', 'OperatorCompensationMethod', 
  'Continent', 'Country', 'Region'
)

dat.cols <- setdiff(names(d), c(idx.cols, meta.cols))

col <- 'ActualHouseholds'
X <- d %>% select(-one_of(col), -one_of(meta.cols))
y <- d[,col]

X.tr <- X[!is.na(y),]
X.ho <- X[is.na(y),]
y.tr <- y[!is.na(y)]
y.ho <- y[is.na(y)]

library(caret)
trc <- trainControl(method='cv', number=10, savePredictions='final')
X.tr.m <- model.frame(~., data=X.tr, ) %>% data.frame %>% select(-one_of('(Intercept)'))
m <- train(X.tr, y.tr, method='gbm', tuneLength=4, trControl = trc, verbose=F)

m$pred %>% ggplot(aes(x=pred, y=obs)) + geom_point()

d.idx <- d %>% select(one_of(idx.cols))
d.dat <- d %>% select(-one_of(idx.cols))


##### Model Training #####

# Initialize training parameters
tr <- proj$getTrainer()
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

# Train models
results <- lapply(models, function(m) tr$train(m, X, y, enable.cache=T)) %>% setNames(names(models))

# Alternatively, all currently cached models can be fetched by running:
# results <- proj$getModels(names.only=F)

##### Performance Evaluation #####

# Plot performance measures using directly attached resampled statistics and save result
resamp <- GetResampleData(results)
GetDefaultPerfPlot(resamp, 'accuracy')
proj$saveResult('perf_resample', resamp)

# Plot performance measures computed on-the-fly (should be identical results in this case)
perf <- GetPerfData(results)
GetDefaultPerfPlot(resamp, 'roc')
proj$saveResult('perf_roc', perf)


##### Inference #####

# Variable Importance
var.imp <- GetVarImp(results)
PlotVarImp(var.imp, compress=F)
proj$saveResult('var_imp', var.imp)

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
proj$saveResult('partial_dependence', pd.data)


