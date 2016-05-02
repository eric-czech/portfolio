source('~/repos/portfolio/functional/common/R/cache.R')
source('~/repos/portfolio/functional/ml/R/trainer.R')

MLProject <- setRefClass(
  "MLProject",
  fields = list(trainer='SimpleTrainer', seed='numeric', proj.dir='character', res.cache='Cache'),
  methods = list(

    #' @title Constructor for project
    #' @param proj.dir path to folder to contain all non-code data for project (e.g.
    #' "~/data/myproject")
    #' @param seed seed to use throughout project
    initialize = function(..., proj.dir, seed){
     
      res.cache <<- Cache(dir=file.path(proj.dir, 'cache'), project='results')
      trainer <<- SimpleTrainer(cache.dir=file.path(proj.dir, 'cache'), cache.project='models', seed=seed)
     
      callSuper(..., trainer=trainer, seed=seed, proj.dir=proj.dir, res.cache=res.cache)
    },
    getTrainer = function(){ trainer },
    getDataDir = function(){
      file.path(proj.dir, 'data')
    },
    getData = function(file.name, ...){
      f <- file.path(getDataDir(), file.name)
      cat(sprintf('Loading raw project data from file "%s"', f))
      read.csv(f, ...)
    },
    getModels = function(filter=NULL, names.only=T){
      
      # Fetch all saved model training results
      model.names <- trainer$getFitModelNames()
      
      # Apply custom filter to model names if given
      if (!is.null(filter))
        model.names <- model.names[sapply(model.names, filter)]
      
      # Return either just the model names or a list of model objects
      # keyed by name, depending on flag passed
      if (names.only) model.names
      else setNames(lapply(model.names, function(m) trainer$getFitModelObject(m)), model.names)
    },
    saveResult = function(name, result){
      res.key <- trainer$cleanObjectName(name)
      res.cache$store(res.key, result)
    },
    loadResult = function(name){
      res.key <- trainer$cleanObjectName(name)
      res.cache$get(res.key)
    }
  )
)