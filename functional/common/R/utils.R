#'-----------------------------------------------------------------------------
#' Miscellaneous Utilities
#'
#' TBD
#' 
#' @author eczech
#'-----------------------------------------------------------------------------

# Import Command:
# library(devtools)
# source_url('https://cdn.rawgit.com/eric-czech/portfolio/master/demonstrative/R/common/utils.R')

coalesce <- function(value, default){
  #' Returns the given value if not null or NA; 
  #' returns #default value otherwise.
  #' 
  #' Note that the inputs may be any equal length vectors.
  if (is.null(value)) default
  else ifelse(is.na(value), default, value)
}

lib <- function(p) {
  #' Requires the given package, installing it first if necessary
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

#' Binary operator used to remove items from a vector
`%wo%` <- function(sequence, removals) sequence[!sequence %in% removals]
