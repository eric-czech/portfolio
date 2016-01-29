#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/utils.R')
#source_url('http://cdn.rawgit.com/eric-czech/portfolio/master/functional/common/R/cache.R')
source('~/repos/portfolio/functional/common/R/cache.R')
library(foreach)
library(iterators)
library(logging)

Trainer <- setRefClass("Trainer",
  fields = list(cache='Cache', seed='numeric', fold.data='list', fold.index='list'),
  methods = list(
    initialize = function(..., cache.dir, cache.project, seed=1){
      cache <<- Cache(dir=cache.dir, project=cache.project)
      callSuper(..., cache=cache, seed=seed, fold.data=list(), fold.index=list())
    },
    getCache = function(){ cache },
    generateFolds = function(fold.generator){
      set.seed(seed)
      fold.index <<- fold.generator()
      loginfo('Fold index generation complete')
    },
    getFolds = function(){ fold.index },
    generateFoldData = function(X, y, data.generator, data.summarizer=NULL){
      if (length(fold.index) == 0)
        stop('Training cannot be done until folds have been generated (must call generateFolds first)')
      fold.data <<- foreach(fold=fold.index, i=icount()) %do%{
        loginfo('Generating data for fold %s of %s', i, length(fold.index))
        
        X.train <- X[-fold,]; y.train <- y[-fold]
        X.test <- X[fold,]; y.test <- y[fold]
        
        set.seed(seed)
        fold.key <- sprintf('fold_%s', i)
        d <- cache$load(fold.key, function(){ data.generator(X.train, y.train, X.test, y.test) })
        
        if (!is.null(data.summarizer)) data.summarizer(d)
        
        list(key=fold.key, id=i, data=d, y.test=y.test)
      }
      loginfo('Fold data generation complete')
    },
    getFoldData = function(){
      fold.data
    },
    train = function(model, data.summarizer=NULL){
      if (length(fold.data) == 0)
        stop('Training cannot be done until fold data has been generated (must call generateFoldData first)')
      res <- foreach(fold=fold.data, i=icount()) %do% {
        loginfo('Running model trainer for fold %s of %s', i, length(fold.data))
        
        if (!is.null(data.summarizer)) data.summarizer(fold$data)
        
        set.seed(seed)
        f <- model$train(fold$data)
        
        set.seed(seed)
        p <- model$predict(f, fold$data)
        
        list(fit=f, y.pred=p, y.test=fold$y.test, fold=fold$id)
      }
      loginfo('Training complete')
      res
    }
  )
)

# t <- Trainer(cache.dir='/tmp', cache.project='test')
