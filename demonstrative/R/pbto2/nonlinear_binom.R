library(foreach)
library(dplyr)
library(ggplot2)
library(rstan)
library(reshape2)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

d <- read.csv('~/data/pbto2/export/data_stan_input.csv', stringsAsFactors=F)
d$rand <- rnorm(n=nrow(d))

#features <- c('pbto2', 'age', 'marshall', 'gcs', 'sex')
static.features <- c('age', 'marshall', 'gcs', 'sex')
ts.feature <- c('icp1')
features <- c(static.features, ts.feature)

d.stan <- get.cleaned.data(d, features, scale=T, sample.frac=NULL, outcome.func=gos.to.binom)
dsu <- get.cleaned.data(d, features, scale=F, sample.frac=NULL, outcome.func=gos.to.binom)
unscaled.value <- function(x, var) x * sd(dsu[,var]) + mean(dsu[,var])

print(paste0('length before = ', nrow(d), ', length after = ', nrow(d.stan)))

if (sum(is.na(d.stan[,ts.feature])) > 0)
  stop('Found na ts values')

### Stan

d.model <- get.stan.data(d.stan, static.features, ts.feature)
setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'nonlinear_binom_2.stan'

posterior <- stan(model.file, data = d.model,
                  warmup = 200, iter = 5000, thin = 30, 
                  chains = 4, verbose = FALSE)

# posterior <- stan(model.file, data = d.model,
#                   warmup = 150, iter = 4000, thin = 5, 
#                   chains = 14, verbose = FALSE)

# Running parallel chains on Mac
library(parallel) # or some other parallelizing package
n.chains <- 5

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 250, iter = 2500, chains = 1, thin = 20, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)


pars <- c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha', 'p')
post <- rstan::extract(posterior)
print(posterior, pars)

rstan::traceplot(posterior, c('beta', 'betaz', 'a1', 'a2', 'b1', 'b2', 'c', 'alpha'))
plot(posterior)

def.off()

x <- seq(min(d.stan[,ts.feature]), max(d.stan[,ts.feature]), length.out = 100)
y.est <- get.mean.curve(post, x)
y.main <- data.frame(i=0, x=unscaled.value(x, ts.feature), y=y.est)
#y.main %>% ggplot(aes(x=x, y=y)) + geom_line()

n = length(post$lp__)
y.samp <- foreach(i=1:n, .combine=rbind) %do% {
  y <- double.logistic(x, post$a1[i], post$a2[i], post$b1[i], post$b2[i], post$c[i, 1], post$c[i, 2])
  a = sum((y - y.est)^2)
  data.frame(i, x=unscaled.value(x, ts.feature), y, a=a)
} %>% mutate(a=(1-scale.minmax(a))^10)

ggplot(NULL) + 
  geom_line(aes(x=x, y=y), color='blue', size=1, data=y.main, alpha=1) + 
  geom_line(aes(x=x, y=y, group=i, alpha=a), data=y.samp) + 
  scale_alpha(range = c(.02, .1)) + theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  xlab(ts.feature) + ylab(paste0('w(', ts.feature, ')')) + ggtitle('Timeseries Weight Function') + 
  #xlim(-10, 175) + 
  ggsave('~/repos/portfolio/demonstrative/R/pbto2/sim/images/wt_actual_pao2.png')


# bins = seq(0, max(fun.sim$y)+1, length.out = 500)
# #bins = seq(0, 100, length.out = 500)
# d.hist <- fun.sim %>% group_by(x) %>% do({
#   h <- hist(.$y, breaks=bins, plot=F)
#   data.frame(x=.$x[1], d=h$density, y=bins[1:(length(bins)-1)])
# })
# d.hist %>% ggplot(aes(x=x, y=y, fill=d)) + geom_tile() + scale_fill_gradient(trans = "log") + ylim(0, 50)
# 
# fun.sim %>% ggplot(aes(x=x, y=y, color=i, group=i)) + geom_line() + scale_y_sqrt()
# fun.sim %>% ggplot(aes(x=factor(x), y=y)) + geom_boxplot() + scale_y_sqrt()







#exp(post$lp)

beta.post <- compute.var.posteriors(d, post)

plot.pbto2.cutoff(beta.post)

plot.beta.post(beta.post)

d.lp <- post$lp %>% melt(id.vars='rowname')
d.lp.iter <- d.lp %>% group_by(iterations) %>% summarise(min=min(value)) %>% 
  #filter(min > -800) %>% 
  .$iterations %>% unique
d.lp %>% 
  filter(iterations %in% d.lp.iter) %>%
  #ggplot(aes(x=factor(Var2), y=value)) + geom_boxplot()
  ggplot(aes(x=Var2, y=value, color=factor(iterations))) + geom_line()

beta.post %>% select(pbto2_cp, pbto2_lo) %>% 
  ggplot(aes(x=pbto2_cp, y=pbto2_lo)) + geom_point()
