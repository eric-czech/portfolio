scale <- function(x) (x - mean(x)) / sd(x)

scale.var <- function(x, d, var) (x - mean(d[,var])) / sd(d[,var])

unscale.var <- function(x, d, var) x * sd(d[,var]) + mean(d[,var])

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

