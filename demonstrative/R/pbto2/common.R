scale <- function(x) (x - mean(x)) / sd(x)

scale.minmax <- function(x) (x - min(x)) / (max(x) - min(x))

scale.var <- function(x, d, var) (x - mean(d[,var])) / sd(d[,var])

unscale.var <- function(x, d, var) x * sd(d[,var]) + mean(d[,var])

gos.to.ord <- function(x){
  if (x >= 4) 3
  else if (x >= 2) 2
  else 1
}

gos.to.binom <- function(x){
  if (x <= 3) 0
  else 1
}

sample.uids <- function(d, frac=1) {
  uids <- sample(unique(d$uid), size = floor(length(unique(d$uid)) * frac), replace = F)
  d %>% filter(uid %in% uids)
}

compute.var.posteriors <- function(d, post){
  beta.post <- data.frame(post$beta) %>% setNames(features[-1]) %>% dplyr::mutate(samp_id=1:nrow(.))
  beta.post$pbto2_cp <- unscale.var(post$z_cutpoint %>% as.numeric, d, 'pbto2')
  beta.post$pbto2_hi <- post$beta_z_hi %>% as.numeric
  beta.post$pbto2_lo <- post$beta_z_lo %>% as.numeric
  if ('t_cutpoint' %in% names(post)){
    beta.post$time_cp <- post$t_cutpoint %>% as.numeric
  }
  beta.post
}

plot.pbto2.cutoff <- function(beta.post){
  beta.post %>% 
    ggplot(aes(x=pbto2_cp)) + geom_histogram(binwidth=1, alpha=.5) + 
    theme_bw() + ggtitle('Pbto2 Cutpoint Estimates') + 
    xlab('Pbto2 Cutoff')
}

plot.time.cutoff <- function(beta.post){
  beta.post$time_cp %>% table %>% melt %>% setNames(c('time_cp', 'ct')) %>% 
    ggplot(aes(x=factor(time_cp), y=ct)) + geom_bar(stat='identity')
}

plot.beta.post <- function(beta.post) {
  beta.post %>% melt(id.vars='samp_id') %>% 
    filter(!variable %in% c('pbto2_cp', 'time_cp')) %>%
    dplyr::group_by(variable) %>% 
    summarise(
      lo=quantile(value, .025), 
      mid=quantile(value, .5), 
      hi=quantile(value, .975)
    ) %>% dplyr::mutate(variable=factor(variable)) %>% 
    ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
    geom_pointrange(size=1) + coord_flip() + theme_bw() + 
    geom_hline(yintercept=0, linetype='dashed') + 
    ggtitle('Coefficient 95% Intervals') + xlab('Coefficient Value') + ylab('Variable')
}

plot.pbto2.coef <- function(beta.post){
  beta.post %>% 
    dplyr::select(pbto2_lo, pbto2_hi, samp_id) %>%
    melt(id.vars='samp_id') %>%
    mutate(variable=ifelse(variable == 'pbto2_lo', 'Pbto2 Below Cutpoint', 'Pbto2 Above Cutpoint')) %>%
    ggplot(aes(x=value, fill=variable)) + geom_density(alpha=.5) +
    theme_bw() + xlab('Coefficient Value') + ylab('Density') +
    ggtitle('Coefficient 95% Intevals for Pbto2 Above and Below Cutpoint') 
}


DATA_CONFIG <- 'config2'
#DATA_CONFIG <- 'config1'

get.wide.data <- function(scale.vars=T, outcome.func=gos.to.binom, reset.uid=F){
  d <- read.csv(sprintf('/Users/eczech/data/pbto2/final/%s/data_wide.csv', DATA_CONFIG)) %>% 
    mutate_each(funs(as.numeric)) %>%
    # Remove pct values in "normal" ranges for blood gases
    dplyr::select(-paco2_35_45, -icp1_0_20, -pha_7.35_7.45, -pao2_30_100, -pbto2_20_100, -starts_with('n_')) %>%
    mutate(gos = sapply(gos, outcome.func))
  
  if (scale.vars)
    d <- d %>% mutate_each(funs(scale), -gos)
  
  if (reset.uid)
    d <- d %>% mutate(uid=as.integer(factor(uid)))
  
  d
}

get.long.data <- function(p, scale.vars=T, outcome.func=gos.to.binom, sample.frac=NULL, reset.uid=T, rm.na=T){
  d <- read.csv(sprintf('~/data/pbto2/final/%s/data_long.csv', DATA_CONFIG), stringsAsFactors=F) %>%
    mutate(gos = sapply(gos, outcome.func))
  
  if (!is.null(sample.frac))
    d <- d %>% sample.uids(frac=sample.frac)
  
  d <- d %>% 
    dplyr::rename(outcome=gos) %>%
    dplyr::select_(.dots=c(p, 'outcome', 'uid')) 
  
  if (rm.na)
    d <- d %>% na.omit()
  
  if (scale.vars)
    d <- d %>% mutate_each_(funs(scale), p)
  
  if (reset.uid)
    d <- d %>% mutate(uid=as.integer(factor(uid)))
  
  d
}


get.wide.model.data <- function(d, features, d.ho=NULL, outcome.var='gos'){
  d <- data.frame(d)
  r <- list(
    N_OBS = nrow(d),
    N_VARS = length(features),
    y = as.integer(d[,outcome.var]),
    x = d[,features]
  )
  if (!is.null(d.ho)){
    r[['N_OBS_HO']] <- nrow(d.ho)
    r[['x_ho']] <- d.ho[,features]
    r[['y_ho']] <- as.integer(d.ho[,outcome.var])
  }
  r
}

