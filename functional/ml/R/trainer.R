#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/utils.R')
#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/cache.R')
source('~/repos/portfolio/functional/common/R/cache.R')
library(logging)
library(stringr)

Trainer <- setRefClass(
  "Trainer",
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
        stop('Training cannot be done before fold index has been generated (must call generateFoldIndex first)')
      fold.data <<- lapply(seq_along(fold.index[['outer']]), function(i){
        fold <- fold.index[['outer']][[i]]
        loginfo('Generating data for fold %s of %s', i, length(fold.index[['outer']]))
        
        # Folds should be for training set, not test set
        X.train <- X[fold,]; y.train <- y[fold]
        X.test <- X[-fold,]; y.test <- y[-fold]
        
        fold.key <- sprintf('fold_%s', i)
        if (!enable.cache) cache$invalidate(fold.key)
        set.seed(seed)
        d <- cache$load(fold.key, function(){ data.generator(X.train, y.train, X.test, y.test) })
        
        if (!is.null(data.summarizer)) data.summarizer(d)
        
        inner.fold.index <- tryCatch(fold.index[['inner']][[i]], error=function(e) NULL)
        list(key=fold.key, id=i, data=d, index=inner.fold.index)
      })
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
      f <- model$train(data, fold.index, fold.id) 
      
      # Reset seed and produce predictions from model
      set.seed(seed)
      p <- model$predict(f, data, fold.id)
      
      list(fit=f, y.pred=p, y.test=model$test(data), fold=fold.id, model=model$name)
    }, 
    holdout = function(models, X, y, X.ho, y.ho, data.generator, cache.key,
                       data.summarizer=NULL, enable.cache=T){
      
      # Create and cache preprocessed predictor data frame
      if (!enable.cache) cache$invalidate(cache.key)
      set.seed(seed)
      d <- cache$load(cache.key, function(){ data.generator(X, y, X.ho, y.ho) })
      
      if (!is.null(data.summarizer)) data.summarizer(d)
      
      res <- lapply(models, function(model){
        loginfo('Creating holdout predictions for model "%s"', model$name)
        .self$predict(model, d, NULL, 0)
      }) %>% setNames(sapply(models, function(m) m$name))
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
        lapply(seq_along(fold.data), function(i){
          fold <- fold.data[[i]]
          loginfo('Running %s model for fold %s of %s', model$name, i, length(fold.data))
          
          if (!is.null(data.summarizer)) data.summarizer(fold$data)
          
          # Fit model and get predictions for this fold
          res <- .self$predict(model, fold$data, fold$index, fold$id)
          
          res
        })
      })
      loginfo('Training complete for model "%s"', model$name)
      res
    }
  )
)

# t <- Trainer(cache.dir='/tmp', cache.project='test')

SimpleTrainer <- setRefClass(
  "SimpleTrainer",
  fields = list(cache='Cache', seed='numeric'),
  methods = list(
    initialize = function(..., cache.dir, cache.project, seed=1){
      cache <<- Cache(dir=cache.dir, project=cache.project)
      callSuper(..., cache=cache, seed=seed)
    },
    cleanModelName = function(model.name){
      str_replace_all(str_replace_all(model.name, '\\.', '_'), '\\W+', '')
    },
    train = function(model, X, y, enable.cache=T){
      model.key <- sprintf('model_%s', cleanModelName(model$name))
      loginfo('Beginning training for model "%s" (cache name = "%s")', model$name, model.key)
      
      if (!enable.cache) cache$invalidate(model.key)
      f <- cache$load(model.key, function(){ 
        set.seed(seed)
        model$train(X, y)
      })
      
      loginfo('Training complete for model "%s"', model$name)
      list(fit=f)
    }
  )
)