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

#' @title Load a library
#' @description  Requires the given package, installing it first if necessary
lib <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

#' @title Imports code in a relative path
#' @description This is helpful for sourcing code when it is necessary
#' to switch between using local versions of that code and versions
#' hosted as static files in CDN (e.g. github's)
import_source <- function(path, root=NULL, return.status=F){
  require(devtools)
  if (is.null(root)){
    if (!is.null(options()$import.root)){
      root <- options()$import.root
    } else {
      stop('Imports must be specified with root path if not set via options(import.root="/path/or/url")')
    }
  }
  source.path <- file.path(root, path)
  res <- try(source(source.path), silent=T)
  if ("try-error" %in% class(res))
    res <- try(source_url(source.path), silent=T)
  res <- "try-error" %in% class(res)
  if (return.status) res
  else invisible(res)
}

#' Binary operator used to remove items from a vector
`%wo%` <- function(sequence, removals) sequence[!sequence %in% removals]
