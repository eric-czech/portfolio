library(glmulti)
library(MASS)
library(dplyr)
select <- dplyr::select
source('~/repos/portfolio/demonstrative/R/pbto2/common.R')

dbu <- get.wide.data(outcome.func=gos.to.binom, scale.vars=F, remove.na.flags=F)
p <- c('age', 'sex', 'marshall', 'gcs')

# Part 1: PbtO2 + Outcome 

## PbtO2 + Outcome (linear models)

- Show coefficients and AIC from covariates + PbtO2 
- Explain significance of PbtO2
- ask about validation of 20 for pbto2
- ask about upper pbto2 cutoff

## PbtO2 + Outcome (exhaustive models)

- Show that PbtO2 is not lost in exhaustive AIC search

## PbtO2 + Outcome (flexible models)

- Show fancy model and threshold chosen for PbtO2 
- Mention what David said about high PbtO2
- Show CV performance (difference of AUC distributions)

# Part 2: PbtO2 + Mortality

- Same as part 1 but with different outcome

# Part 3: PbtO2 vs PaO2

## Linear Models

- Show glm results from the following:
  - PaO2 + Pbto2 all samples
  - PaO2 + Pbto2 only samples w/ icp
  - PaO2 + Pbto2 + ICP

## ROC Analysis

- Show ROC curves and AUC numbers for models containing all covariates + PbtO2 vs the same w/ PaO2
- Also show the same as above with ICP


