#'-----------------------------------------------------------------------------
#' Data Generating Scripts for ML Templates
#'
#' This module contains code for generating sample data to test training 
#' routines on.
#' 
#' @author eczech
#'-----------------------------------------------------------------------------

library(caret)
PROJ_DIR <- '~/data/unit_tests/trainer'

write.sample.data <- function(){
  d <- twoClassSim()
  write.csv(d, file=file.path(PROJ_DIR, 'data', 'sim.csv'), row.names=F)  
}
