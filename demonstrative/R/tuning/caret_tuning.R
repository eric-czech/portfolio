
# SVM tuning example
model.svm.grid <- list(
  train=function(d){
    registerDoMC(3)
    svm.sigma <- GetSvmSigma(d$X.train.sml)
    train(
      d$X.train.sml, d$y.train, method='svmRadial', 
      tuneGrid = expand.grid(.sigma = as.numeric(svm.sigma[-2]), .C = 2^(1:6)),
      trControl = trctrl(verboseIter=T)
    )
  }, predict=predict.test.data
)
fit.svm.grid <- trainer$train(model.svm.grid)