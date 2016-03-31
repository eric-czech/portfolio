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

#' @title Removes the '.Environment' attribute from a modeling result
#' @description This is often necessary because some libraries return 
#' model objects that somewhat inexplicably large in memory and on disk.  The 
#' culprit here seems to often be a hidden attribute attached to the 'terms'
#' and 'formula' attributes in those objects called '.Environment'.  See here
#' for an example: http://stackoverflow.com/questions/29481331/r-attribute-environment-consuming-large-amounts-of-ram-in-nnet-package
#' This can be fixed by setting that attribute to an empty vector (See
#' http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/)
#' @note Note that verifying the size of objects in memory using object.size is very
#' unreliable.  A much more accurate way is \code{length(serialize(object, NULL))}
#' @param model to remove unncessary and often large attributes from
#' @return trimmed model result
trim_model <- function(model){
  try({
    m <- model
    is.caret <- 'finalModel' %in% names(model)
    
    if (is.caret) m <- model$finalModel
    
    # Trim nested glm objects in MARS models
    if (length(class(m)) == 1 && class(m) == 'earth'){
      for (i in seq_along(m$glm.list)) 
        m$glm.list[[i]] <- trim_model(m$glm.list[[i]])
    }
    
    # For every known inner object containing large, unnecessary objects
    for (name in c('terms', 'formula', 'model')){
      # If the model does not contain this attribute, continue
      if (!name %in% names(m))
        next
        
      # Otherwise, remove all known large attributes
      att <-'.Environment'
      if (att %in% names(attributes(m[[name]])) && class(attr(m[[name]], att)) == 'environment'){
        attr(m[[name]], att) <- c() 
      }
      
      # Special case for environments nested within terms object nested within a model object (MARS only so far)
      if (name == 'model' && 'terms' %in% names(attributes(m[[name]]))){
        attr(attr(m[[name]], 'terms'), '.Environment') <- c()
      }
    }
    
    if (is.caret) model$finalModel <- m
    else model <- m
  }, silent=F)
  model
}
