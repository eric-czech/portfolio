#'-----------------------------------------------------------------------------
#' Data Caching Class
#'
#' This class is meant to be used as a convenience for organizing and retrieving
#' large or epensive objects on disk.  Example usage:
#' 
#' basicConfig(level=loglevels['DEBUG']) # Cache messages all made @ DEBUG 
#' cache <- Cache(dir='/tmp', project='myproject') # Objects stored at /tmp/myproject
#' cache$invalidate('myobj') # Remove object from cache dir if it exists
#' 
#' # The following call will fetch the object by name if it already exists;
#' # otherwise the "loader" function will be called and the result from that
#' # function will be cached and returned
#' d <- cache$load('myobj', loader=function(){data.frame(x=1:10)})
#' 
#' cache$disable() # This will cause cache to never load objects 
#'                 # from disk but still write them
#' cache$bypass()  # This will never load or write objects to disk
#' cache$enable()  # This will restore default behavior if either of the above were set
#' 
#' # This would download a url to file in the cache by the given key name
#' # and then continually return the path to that file on future calls, 
#' # rather than returning any deserialized version of that content
#' html.file.path <- c$download('google.html', 'http://www.google.com') 
#' 
#' @author eczech
#'-----------------------------------------------------------------------------
library(logging)

Cache <- setRefClass("Cache",
  fields = list(dir = "character", project = "character", mode = "character"),
  methods = list(
    initialize = function(..., dir=file.path('~', '.Rcache'), project='default', mode='enabled'){
        callSuper(..., dir=dir, project=project, mode=mode)
    },
    normalizeKey = function(key, ext){ 
      if (is.null(ext)) key
      else paste0(key, ext) 
    },
    enable = function(){ mode <<- 'enabled' },
    disable = function(){ mode <<- 'disabled' },
    bypass = function(){ mode <<- 'bypass' },
    setMode = function(x){ 
      x <- tolower(x)
      if (!x %in% c('enabled', 'disabled', 'bypass'))
        stop(sprintf('Cache model "%s" is not valid'))
      mode <<- x 
    },
    getPath = function(key, ext='.Rdata') {
      #' Returns a file path to the object in cache (rooted at dir/project)
      
      # Add extension to object key
      key <- normalizeKey(key, ext)
      
      # Create a path for the parent directory of the object
      cpath <- file.path(dir, project)
      
      # Verify that the parent object already exists or that it could be created
      if (!file.exists(cpath))
        dir.create(cpath, recursive=T)
      if (!file.exists(cpath))
        stop(sprintf('Failed to create or validate cache directory "%s%', cpath))
      
      # Verify that the cached object file can be created if it does not 
      # already exist (ie make sure there are no FS or permission errors)
      cpath <- file.path(cpath, key)
      if (!file.exists(cpath)){
        valid <- file.create(cpath)
        if (!valid) stop(sprintf('Unable to create cached object file at path "%s"', cpath))
        if (!file.remove(cpath)) stop(sprintf('Failed to delete cache file "%s"', cpath))
      }
      
      # Return the full path to the cached object
      cpath
    },
    load = function(key, loader, compress=T, ext='.Rdata'){
      #' Loads an object by the given name (ie key) if it exists or
      #' creates this object using the given loader function and then
      #' returns it (after saving it for later, faster retrieval)
      
      # If the cache is to be completely bypassed, return the result immediately
      if (mode == 'bypass')
        return(loader())
      
      # Get the full path to the cached object
      cpath <- getPath(key, ext)
      
      # Load and save the object if it is not already cached
      if (!file.exists(cpath) || mode != 'enabled'){
        logdebug('Creating and storing cache object "%s"', cpath)
        object <- loader()
        base::save(object, file=cpath, compress=compress)
      } else {
        logdebug('Restoring cached object "%s" from disk', cpath)
      }
      
      # Load and return the cached object
      e <- new.env()
      base::load(cpath, envir=e)
      if ('res' %in% ls(e)){
        object <- e$res
        base::save(object, file=cpath, compress=compress)
      } else {
        if (!'object' %in% ls(e))
          stop(sprintf('Failed to find result with name "object" in restored cache file "%s"', cpath))
        object <- e$object
      }
      e$object
    },
    invalidate = function(key, ext='.Rdata'){
      #' Deletes a file for the given key, if it exists in cache
      cpath <- getPath(key, ext)
      logdebug('Invalidating cached object at path "%s"', cpath)
      if (file.exists(cpath))
        if (!file.remove(cpath)) 
          stop(sprintf('Failed to remove cached object at path "%s"', cpath))
    },
    download = function(key, url, ext=NULL, ...){
      cpath <- getPath(key, ext)
      if (!file.exists(cpath))
        download.file(url, cpath, ...)
      cpath
    },
    store = function(key, object, compress=T, ext='.Rdata'){
      #' Stores the given object using the given name (key)
      #' * Note that this should be used only to force a write of a 
      #'   new cached object, which should be a somewhat rare use case
      cpath <- getPath(key, ext)
      base::save(object, file=cpath, compress=compress)
    },
    get = function(key, ext='.Rdata'){
      #' Fetches a previously cached object using the given name (key)
      cpath <- getPath(key, ext)
      if (!file.exists(cpath))
        stop(sprintf('Cached object "%s" does not exist (expected at path "%s")', key, cpath))
      
      e <- new.env()
      base::load(cpath, envir=e)
      if (!'object' %in% ls(e))
        stop(sprintf('Failed to find result with name "object" in restored cache file "%s"', cpath))
      e$object
    }
  )
)

# Example Usage:
# basicConfig(level=loglevels['DEBUG'])
# c <- Cache(dir='/tmp', project='test2')
# key <- 'test'
# c$disable()
# c$invalidate(key)
# d <- c$load('test', function(){data.frame(x=1:10)})
# d <- c$load('test', function(){data.frame(x=1:10)})