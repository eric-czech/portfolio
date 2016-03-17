#'-----------------------------------------------------------------------------
#' Multicore Utilities
#' 
#' @author eczech
#'-----------------------------------------------------------------------------
library(doParallel)
CLUSTER <- NULL

#' @title Establish parallel backend with given number of concurrent processes
#' @param n.core number of parallel backends to register
#' @param log.file path to file in which output from backends will be written
#' @note IMPORTANT - Do not source this the file containing this function 
#' multiple times.  A cluster object is created each time and if redefined,
#' the cluster associated with the existing backends will be lost (and those
#' backends will be orphaned requiring the processes to be killed manually).
registerCores <- function(n.core, log.file='/tmp/train.log'){
  if (!is.null(CLUSTER)) try(stopCluster(CLUSTER), silent = T)
  CLUSTER <<- makeCluster(n.core, type='PSOCK', outfile=log.file)
  registerDoParallel(CLUSTER)
}