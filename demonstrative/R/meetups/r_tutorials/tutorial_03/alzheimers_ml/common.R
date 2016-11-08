#'-----------------------------------------------------------------------------
#' ML Project Utilities Template
#'
#' This module contains common imports and shared settings to be used 
#' across project modules
#' 
#' @author eczech
#'-----------------------------------------------------------------------------

# Base Imports
library(caret)
library(xgboost)
library(randomForest)
library(glmnet)
library(rpart)

library(plyr); library(dplyr) # This must occur first to avoid later namespace collisions
source('~/repos/portfolio/functional/common/R/utils.R')

options(import.root='~/repos')
import_source('portfolio/functional/ml/R/project.R')
import_source('portfolio/functional/ml/R/results.R')
import_source('portfolio/functional/ml/R/parallel.R')
import_source('portfolio/functional/ml/R/partial_dependence.R')
import_source('portfolio/functional/ml/R/average_ensemble.R')

# Imports for modeling project
library(logging)
basicConfig(loglevels[['DEBUG']]) # Initialize logging

# Pick a seed
SEED <- 383

# Initialize project manager
proj <- MLProject(proj.dir='~/data/meetups/r_tutorials/tutorial_03', seed=SEED)
