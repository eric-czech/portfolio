library(foreach)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(rstan)
library(reshape2)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_binom_utils.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

static.features <- c('age', 'marshall', 'gcs', 'sex')
ts.feature <- c('pao2')
features <- c(static.features, ts.feature)

dsu <- get.long.data(features, scale.vars=F, outcome.func=gos.to.binom, reset.uid=T)
dsu$rand <- rnorm(n = nrow(dsu))
unscaled.value <- function(x, var) x * sd(dsu[,var]) + mean(dsu[,var])
d.stan <- dsu %>% mutate_each_(funs(scale), features)

if (sum(is.na(d.stan[,ts.feature])) > 0)
  stop('Found na ts values')

### Stan

d.model <- get.stan.data(d.stan, static.features, ts.feature)
setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'nonlinear_slogit_binom.stan'

posterior <- stan(model.file, data = d.model,
                  warmup = 200, iter = 5000, thin = 30, 
                  chains = 1, verbose = FALSE)

# posterior <- stan(model.file, data = d.model,
#                   warmup = 150, iter = 4000, thin = 5, 
#                   chains = 14, verbose = FALSE)

# Running parallel chains on Mac

library(parallel) # or some other parallelizing package
n.chains <- 5

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 300, iter = 3000, chains = 1, thin = 3, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)


pars <- c('beta', 'betaz', 'b', 'c', 'alpha')
post <- rstan::extract(posterior)
print(posterior, pars)

rstan::traceplot(posterior, pars)
plot(posterior)



x <- seq(min(d.stan[,ts.feature]), max(d.stan[,ts.feature]), length.out = 100)
x.unscaled <- unscaled.value(x, ts.feature)

y.est.mean <- get.slogit.mean.curve(post, x, agg.func=mean)
y.est.median <- get.slogit.mean.curve(post, x, agg.func=median)
y.mean <- data.frame(i=0, x=x.unscaled, y=y.est.mean)
y.median <- data.frame(i=0, x=x.unscaled, y=y.est.median)

#y.main %>% ggplot(aes(x=x, y=y)) + geom_line()

n = length(post$lp__)
y.samp <- foreach(i=1:n, .combine=rbind) %do% {
  y <- single.logistic(x, post$betaz[i], post$b[i], post$c[i])
  a = sum((y - y.est.mean)^2)
  data.frame(i, x=unscaled.value(x, ts.feature), y, a=a)
} %>% mutate(a=(1-scale.minmax(a))^10)

v.hist <- hist(dsu[,ts.feature], plot=F, breaks=length(x))
v.width <- v.hist$mids[1] - v.hist$breaks[1]
min.v <- min(min(y.mean$y), min(y.samp$y))
max.v <- max(max(y.mean$y), max(y.samp$y))
v.hist <- data.frame(x=v.hist$mids, y=v.hist$counts/sum(v.hist$counts))
v.hist$y = min.v + .35 * abs(max.v - min.v) * scale.minmax(v.hist$y)

c.mid <- median(post$c) %>% unscaled.value(ts.feature)

p1 <- ggplot(NULL) + 
  geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
  geom_line(aes(x=x, y=y, color='mean'), size=1, data=y.mean, alpha=.75) + 
  geom_line(aes(x=x, y=y, color='median'), size=1, data=y.median, alpha=.75) + 
  scale_alpha(range = c(.05, .05), guide = 'none') + theme_bw() +
  scale_color_discrete(guide = guide_legend(title = "Summary")) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  xlab(ts.feature) + ylab(paste0('w(', ts.feature, ')')) + ggtitle('Timeseries Weight Function') + 
  geom_rect(aes(xmax=x+v.width, xmin=x-v.width, ymin=min.v, ymax=y), data=v.hist, alpha=.5) +
  geom_vline(xintercept=c.mid, linetype='dashed', alpha=.25) +
  annotate("text", x = c.mid, y = 1, label = round(c.mid, 2)) 


post.summary <- get.slogit.posterior.summary(post, static.features)
p2 <- post.summary %>% 
  filter(!variable %in% c('center', 'weight_magnitude', 'intercept')) %>%
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed') + 
  ggtitle('Coefficient 95% Credible Intervals') +
  ylab('Coefficient Range') + xlab('')


# Save above plots to file
file <- sprintf("~/repos/portfolio/demonstrative/R/pbto2/presentations/images/slogit_no_interp/single_var/actual_%s.png", ts.feature)
png(file = file, width=800, height=800)
grid.arrange(p2, p1, nrow=2, ncol=1, heights=c(0.3, 0.7))
dev.off()


