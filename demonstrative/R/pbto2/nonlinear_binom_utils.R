
get.posterior.summary <- function(post, static.features){
  data.frame(
    1:length(post$alpha),
    as.numeric(post$alpha), post$beta[,1], post$beta[,2],
    post$beta[,3], post$beta[,4]
  ) %>% setNames(c(
    'iteration', 'intercept', static.features[1], static.features[2], 
    static.features[3], static.features[4]
  )) %>% melt(id.vars='iteration') %>% group_by(variable) %>% do({
    data.frame(
      lo=quantile(.$value, probs = .025), 
      mid=quantile(.$value, probs = .5), 
      hi=quantile(.$value, probs = .975)
    )
  })
}

get.ts.post.plot <- function(post, vars, x, ts.feature, ts.value){
  
  unscaled.value <- function(x) sd(ts.value, na.rm=T) * x + mean(ts.value, na.rm=T)
  x.unscaled <- unscaled.value(x)
  
  get.curve <- function(post, x, vars, agg.func=median){
    a1 = agg.func(post[[vars['a1']]])
    a2 = agg.func(post[[vars['a2']]])
    b1 = agg.func(post[[vars['b1']]])
    b2 = agg.func(post[[vars['b2']]])
    c1 = agg.func(post[[vars['c1']]])
    c2 = agg.func(post[[vars['c2']]])
    double.logistic(x, a1, a2, b1, b2, c1, c2)
  }
  
  y.est.mean <- get.curve(post, x, vars, agg.func=mean)
  y.est.median <- get.curve(post, x, vars, agg.func=median)
  y.mean <- data.frame(i=0, x=x.unscaled, y=y.est.mean)
  y.median <- data.frame(i=0, x=x.unscaled, y=y.est.median)
  
  n = length(post$lp__)
  y.samp <- foreach(i=1:n, .combine=rbind) %do% {
    a1 <- post[[vars['a1']]]
    a2 <- post[[vars['a2']]]
    b1 <- post[[vars['b1']]]
    b2 <- post[[vars['b2']]]
    c1 <- post[[vars['c1']]]
    c2 <- post[[vars['c2']]]
    y <- double.logistic(x, a1[i], a2[i], b1[i], b2[i], c1[i], c2[i])
    a = sum((y - y.est.mean)^2)
    data.frame(i, x=x.unscaled, y, a=a)
  } %>% mutate(a=(1-scale.minmax(a))^10)
  
  v.hist <- hist(ts.value, plot=F, breaks=length(x))
  v.width <- (v.hist$mids[1] - v.hist$breaks[1])
  min.v <- min(min(y.mean$y), min(y.samp$y))
  max.v <- max(max(y.mean$y), max(y.samp$y))
  v.hist <- data.frame(x=v.hist$mids, y=v.hist$counts/sum(v.hist$counts)) %>%
    mutate(y=min.v + .35 * abs(max.v - min.v) * scale.minmax(y)) %>%
    mutate(xmin=x - v.width, xmax=x + v.width)
  
  c.lo <- median(post[[vars['c1']]]) %>% unscaled.value
  c.hi <- median(post[[vars['c2']]]) %>% unscaled.value
  
  ggplot(NULL) + 
    geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
    geom_line(aes(x=x, y=y, color='mean'), size=1, data=y.mean, alpha=.75) + 
    geom_line(aes(x=x, y=y, color='median'), size=1, data=y.median, alpha=.75) + 
    scale_alpha(range = c(.05, .05), guide = 'none') + theme_bw() +
    scale_color_discrete(guide = guide_legend(title = "Summary")) + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    xlab(ts.feature) + ylab(paste0('w(', ts.feature, ')')) + ggtitle('Timeseries Weight Function') + 
    geom_rect(aes(xmax=xmax, xmin=xmin, ymax=y), ymin=min.v, data=v.hist, alpha=.5) +
    geom_vline(xintercept=c.lo, linetype='dashed', alpha=.25) +
    annotate("text", x = c.lo, y = 1, label = round(c.lo, 2)) + 
    geom_vline(xintercept=c.hi, linetype='dashed', alpha=.25) + 
    annotate("text", x = c.hi, y = 1, label = round(c.hi, 2))
}

