library(foreach)
library(caret)

partialDependence <- function(
  object, X, predictor, pred.fun, 
  verbose = T, grid.size=10, grid.window=c(0, 1), sample.rate=1, 
  allow.parallel = T){
  
  if (missing(object)) stop('A model object must be supplied')
  if (missing(X)) stop('Training data must be supplied')
  if (missing(predictor)) stop('A target predictor name must be supplied')
  if (!is.character(predictor) || length(predictor) > 1) stop('Predictor must be a single string')
  if (missing(pred.fun)) stop('A prediction function must be supplied')
  if (sample.rate <= 0 || sample.rate > 1) stop('Sample rate must be in (0, 1]')
  
  if (verbose) cat(sprintf('PDP: Computing grid points for predictor "%s"\n', predictor))
  
  # Create stratified sample wrt to predictor, if sample.rate present
  index <- NULL
  if (sample.rate < 1){
    N <- nrow(X)
    
    # Stratify predictor values
    partitions <- createFolds(X[, predictor], k=10, returnTrain=F)
    
    # Determine observation sample to use
    sample.fun <- function(p) sample(p, floor(length(p)*sample.rate), replace=F)
    index <- unname(unlist(lapply(partitions, sample.fun)))
    if (any(duplicated(index))) 
      stop('Sampling index includes duplicates?  This should not be possible.')
    
    # Create sampled predictor data frame
    X <- X[index, ]
    
    if (verbose){
      cat(sprintf(
        "PDP: Original predictor dataset reduced to size %s from %s (at sample rate=%s)\n", 
        nrow(X), N, sample.rate
      ))
    }
  }
  
  # Generate predictor value grid to compute PDP for
  N <- nrow(X)
  x <- X[, predictor]
  if (is.factor(x) || length(unique(x)) <= 2){
    grid <- unique(x)
  } else {
    grid <- as.numeric(x)
    x.range <- quantile(x, probs=grid.window)
    grid <- seq(x.range[1], x.range[2], len = grid.size)
  }
  
  # Set number of grid points to compute pdp for
  n.grid <- length(grid)
  if (n.grid == 1){
    stop(sprintf(
      'Predictor "%s" only has one unique value.  Partial dependence calculation requires at least 2', 
      predictor
    ))
  }
  
  if (verbose) {
    cat(sprintf(
      "PDP: Computing PDP estimates for %s predictor values\n", n.grid
    ))
  }
  
  # Determine foreach parallel mode (only use parallel if enabled and multiple backends registered)
  `%op%` <- if (allow.parallel && getDoParWorkers() > 1) `%dopar%` else `%do%`
  
  # Milestone value for progress reporting
  n.split <- floor(n.grid/10)
  
  pdp <- foreach(i=1:n.grid, .combine=rbind) %op% {
    if (verbose && n.split > 0 && i %% n.split == 0) {
      cat(sprintf(
        "PDP: Processing predictor grid point %s of %s (%s%% complete)\n", 
        i, n.grid, round(100*i/n.grid)
      ))
    }
    X[, predictor] <- grid[i]
    pred.fun(object, X)
  }
  rownames(pdp) = grid
  
  list(pdp=pdp, grid=grid, index=index, x=x, predictor=predictor)
}