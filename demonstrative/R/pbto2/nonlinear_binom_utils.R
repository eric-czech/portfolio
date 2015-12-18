
get.posterior.summary <- function(post, static.features){
  data.frame(
    1:length(post$alpha),
    as.numeric(post$alpha), post$beta[,1], post$beta[,2],
    post$beta[,3], post$beta[,4], as.numeric(post$betaz),
    as.numeric(post$c1), as.numeric(post$c2)
  ) %>% setNames(c(
    'iteration', 'intercept', static.features[1], static.features[2], 
    static.features[3], static.features[4], 'weight_magnitude',
    'lower_center', 'upper_center'
  )) %>% melt(id.vars='iteration') %>% group_by(variable) %>% do({
    data.frame(
      lo=quantile(.$value, probs = .025), 
      mid=quantile(.$value, probs = .5), 
      hi=quantile(.$value, probs = .975)
    )
  })
}

get.slogit.posterior.summary <- function(post, static.features){
  data.frame(
    1:length(post$alpha),
    as.numeric(post$alpha), post$beta[,1], post$beta[,2],
    post$beta[,3], post$beta[,4], as.numeric(post$betaz),
    as.numeric(post$c)
  ) %>% setNames(c(
    'iteration', 'intercept', static.features[1], static.features[2], 
    static.features[3], static.features[4], 'weight_magnitude',
    'center'
  )) %>% melt(id.vars='iteration') %>% group_by(variable) %>% do({
    data.frame(
      lo=quantile(.$value, probs = .025), 
      mid=quantile(.$value, probs = .5), 
      hi=quantile(.$value, probs = .975)
    )
  })
}