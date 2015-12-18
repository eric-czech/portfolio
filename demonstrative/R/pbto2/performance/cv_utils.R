library(foreach)
library(loo)
library(stringr)
library(gridExtra)

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
      extract(r$ms.fl, 'Single Logistic'),
      extract(r$mw.fl, 'Wide'),
      extract(r$mn.fl, 'Null')
    )
  }
}

extract.rhat <- function(res){
  extract <- function(stanfit, model, type){
    if (is.null(stanfit))
      return(NULL)
    rhat <- summary(stanfit)$summary[,'Rhat']
    data.frame(rhat) %>% add_rownames('variable') %>% 
      mutate(model.name=model, model.type=type)
  }
  foreach(r=res, .combine=rbind)%do%{
    rbind(
      extract(r$ml.fl, 'Double Logistic', 'Full'),
      extract(r$ml.cv, 'Double Logistic', 'CV'),
      extract(r$ms.fl, 'Single Logistic', 'Full'),
      extract(r$ms.cv, 'Single Logistic', 'CV'),
      extract(r$mw.fl, 'Wide', 'Full'),
      extract(r$mw.cv, 'Wide', 'CV'),
      extract(r$mn.fl, 'Null', 'Full'),
      extract(r$mn.cv, 'Null', 'CV')
    ) %>% mutate(fold=r$fold)
  }
}

plot.rhat <- function(rhat){
  plot.group.rhat <- function(r)
    ggplot(r, aes(x=rhat)) + geom_histogram(binwidth=.01) + facet_grid(fold~model.name, scales='free_y')
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
      extract(r$ds, r$ms.cv, 'Single Logistic'),
      extract(r$dw, r$mw.cv, 'Wide'),
      extract(r$dn, r$mn.cv, 'Null')
    ) %>% mutate(fold=r$fold)
  }
}