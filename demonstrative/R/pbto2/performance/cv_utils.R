library(foreach)
library(loo)
library(stringr)
library(gridExtra)

CV_RES_DIR <- '~/data/pbto2/cache' 

get.cv.res.file <- function(ts.feature){
  sprintf('%s/cv_res_%s.Rdata', CV_RES_DIR, ts.feature)
}

get.cv.res.data <- function(ts.feature){
  res.env <- new.env()
  res.file <- get.cv.res.file(ts.feature)
  load(res.file, envir=res.env)
  res.env$res
}

get.cv.res.features <- function(){
  foreach(f=list.files(CV_RES_DIR, pattern="^cv_res_.*\\.Rdata$"), .combine=c)%do%{
    p <- str_split(f, '_')
    if (length(p) != 1 || length(p[[1]]) != 3)
      return(NULL)
    p <- str_split(p[[1]][3], '\\.')
    if (length(p) != 1 || length(p[[1]]) != 2)
      return(NULL)
    p[[1]][1]
  }
}
  
extract.waic <- function(res){
  foreach(r=res, .combine=rbind) %do% {
    if (r$fold != 1)
      return(NULL)
    #browser()
    extract <- function(stanfit, model){
      w <- waic(extract_log_lik(stanfit))
      data.frame(model=model, waic=w$waic, waic_se=w$se_waic, p=w$p_waic, p_se=w$se_p_waic)
    }
    rbind(
      extract(r$ml.fl, 'Double Logistic'),
      extract(r$msc.fl, 'Centered Single Logistic'),
      extract(r$msl.fl, 'Lower Single Logistic'),
      extract(r$msh.fl, 'Upper Single Logistic'),
      extract(r$mw.fl, 'Wide'),
      extract(r$mn.fl, 'Null')
    )
  }
}

extract.rhat <- function(res){
  extract <- function(stanfit, model, type){
    if (is.null(stanfit)) return(NULL)
    rhat <- summary(stanfit)$summary[,'Rhat']
    data.frame(rhat) %>% add_rownames('variable') %>% 
      mutate(model.name=model, model.type=type)
  }
  foreach(r=res, .combine=rbind)%do%{
    rbind(
      extract(r$ml.fl, 'Double Logistic', 'Full'),
      extract(r$ml.cv, 'Double Logistic', 'CV'),
      extract(r$msc.fl, 'Centered Single Logistic', 'Full'),
      extract(r$msc.cv, 'Centered Single Logistic', 'CV'),
      extract(r$msl.fl, 'Lower Single Logistic', 'Full'),
      extract(r$msl.cv, 'Lower Single Logistic', 'CV'),
      extract(r$msh.fl, 'Upper Single Logistic', 'Full'),
      extract(r$msh.cv, 'Upper Single Logistic', 'CV'),
      extract(r$mw.fl, 'Wide', 'Full'),
      extract(r$mw.cv, 'Wide', 'CV'),
      extract(r$mn.fl, 'Null', 'Full'),
      extract(r$mn.cv, 'Null', 'CV')
    ) %>% mutate(fold=r$fold.i)
  }
}

plot.rhat <- function(rhat){
  plot.group.rhat <- function(r)
    ggplot(r, aes(x=rhat)) + geom_histogram(binwidth=.005) + 
    facet_grid(fold~model.name, scales='free_y')
  p1 <- rhat %>% filter(model.type == 'Full') %>% plot.group.rhat
  p2 <- rhat %>% filter(model.type == 'CV') %>% plot.group.rhat
  grid.arrange(p1, p2, ncol=1)
}

extract.predictions <- function(res){
  extract <- function(data, stanfit, model){
    post <- rstan::extract(stanfit)
    data.frame(model=model, y.pred=post$y_pred %>% apply(2, mean), y.true=data$y_ho)
  }
  foreach(r=res, .combine=rbind)%do%{
    rbind(
      extract(r$dl, r$ml.cv, 'Double Logistic'),
      extract(r$dsc, r$msc.cv, 'Centered Single Logistic'),
      extract(r$dsl, r$msl.cv, 'Lower Single Logistic'),
      extract(r$dsh, r$msh.cv, 'Upper Single Logistic'),
      extract(r$dw, r$mw.cv, 'Wide'),
      extract(r$dn, r$mn.cv, 'Null')
    ) %>% mutate(fold=r$fold)
  }
}

# Cross validation performance functions


logloss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}
compute.fold.logloss <- function(preds){
  preds %>% group_by(model, fold) %>% do({
    data.frame(logloss=logloss(.$y.true, .$y.pred))
  })
}

compute.fold.auc <- function(preds){
  preds %>% group_by(model, fold) %>% do({
    p <- prediction(.$y.pred, .$y.true)
    auc <- p %>% performance('auc') %>% .@y.values
    data.frame(auc=auc[[1]])
  })
}