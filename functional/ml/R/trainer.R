#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/utils.R')
#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/cache.R')
source('~/repos/portfolio/functional/common/R/cache.R')
library(foreach)
library(iterators)
library(logging)
library(stringr)

Trainer <- setRefClass("Trainer",
  fields = list(cache='Cache', seed='numeric', fold.data='list', fold.index='list'),
  methods = list(
    initialize = function(..., cache.dir, cache.project, seed=1){
      cache <<- Cache(dir=cache.dir, project=cache.project)
      callSuper(..., cache=cache, seed=seed, fold.data=list(), fold.index=list())
    },
    getCache = function(){ cache },
    generateFoldIndex = function(y, fold.generator){
      set.seed(seed)
      fold.index <<- list()
      fold.index[['outer']] <<- fold.generator(y, index=NULL, level=1)
      fold.index[['inner']] <<- lapply(fold.index[['outer']], function(x) fold.generator(y, index=x, level=2))
      loginfo('Fold index generation complete')
    },
    getFoldIndex = function(){ fold.index },
    generateFoldData = function(X, y, data.generator, data.summarizer=NULL, enable.cache=T){
      if (length(fold.index[['outer']]) == 0)
        stop('Training cannot be done until fold index has been generated (must call generateFoldIndex first)')
      fold.data <<- foreach(fold=fold.index[['outer']], i=icount(), .errorhandling='stop') %do%{
        loginfo('Generating data for fold %s of %s', i, length(fold.index[['outer']]))
        
        # Folds should be for training set, not test set
        X.train <- X[fold,]; y.train <- y[fold]
        X.test <- X[-fold,]; y.test <- y[-fold]
        
        set.seed(seed)
        fold.key <- sprintf('fold_%s', i)
        tryCatch({
          if (!enable.cache) cache$invalidate(fold.key)
          d <- cache$load(fold.key, function(){ data.generator(X.train, y.train, X.test, y.test) })
        }, error=function(e) {
            logerror('An error occurred while generating/fetching training data.')
            logerror(e)
            browser()
        })
        
        if (!is.null(data.summarizer)) data.summarizer(d)
        
        inner.fold.index <- tryCatch(fold.index[['inner']][[i]], error=function(e) NULL)
        list(key=fold.key, id=i, data=d, y.test=y.test, index=inner.fold.index)
      }
      loginfo('Fold data generation complete')
    },
    getFoldData = function(){
      fold.data
    },
    cleanModelName = function(model.name){
      str_replace_all(str_replace_all(model.name, '\\.', '_'), '\\W+', '')
    },
    predict = function(model, data, fold.index, fold.id){
      
      # Run model training routine
      set.seed(seed)
      tryCatch({ 
        f <- model$train(data, fold.index, fold.id) 
      }, error=function(e){
        logerror('An error occurred during training.')
        logerror(e)
        browser()
      })
      
      # Reset seed and produce predictions from model
      set.seed(seed)
      tryCatch({ 
        p <- model$predict(f, data, fold.id)
      }, error=function(e){
        logerror('An error occurred during model prediction')
        logerror(e)
        browser()
      })
      
      list(fit=f, y.pred=p, y.test=model$test(data), fold=fold.id, model=model$name)
    }, 
    holdout = function(models, X, y, X.ho, y.ho, data.generator, data.summarizer=NULL, enable.cache=T){
      
      # Create and cache preprocessed predictor data frame
      set.seed(seed)
      data.key <- 'holdout_data'
      tryCatch({
        if (!enable.cache) cache$invalidate(data.key)
        d <- cache$load(data.key, function(){ data.generator(X, y, X.ho, y.ho) })
      }, error=function(e) {
        logerror('An error occurred while creating/fetching hold out data.')
        logerror(e)
        browser()
      })
      
      if (!is.null(data.summarizer)) data.summarizer(d)
      
      res <- foreach(model=models) %do% {
        loginfo('Creating holdout predictions for model "%s"', model$name)
        .self$predict(model, d, NULL, 0)
      } %>% setNames(sapply(models), function(m) m$name)
      loginfo('Holdout predictions complete')
      res
    },
    train = function(model, data.summarizer=NULL, enable.cache=T){
      if (length(fold.data) == 0)
        stop('Training cannot be done until fold data has been generated (must call generateFoldData first)')
      
      model.key <- sprintf('model_%s', cleanModelName(model$name))
      loginfo('Beginning training for model "%s" (cache name = "%s")', model$name, model.key)
      
      if (!enable.cache) cache$invalidate(model.key)
      res <- cache$load(model.key, function(){ 
        foreach(fold=fold.data, i=icount(), .errorhandling='stop') %do% {
          loginfo('Running model trainer for fold %s of %s', i, length(fold.data))
          
          if (!is.null(data.summarizer)) data.summarizer(fold$data)
          
          # Fit model and get predictions for this fold
          res <- .self$predict(model, fold$data, fold$index, fold$id)

          res
        }
      })
      loginfo('Training complete for model "%s"', model$name)
      res
    }
  )
)

# t <- Trainer(cache.dir='/tmp', cache.project='test')
