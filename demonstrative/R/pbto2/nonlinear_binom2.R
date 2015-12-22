library(foreach)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(rstan)
library(reshape2)
library(loo)

source('~/repos/portfolio/demonstrative/R/pbto2/common.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_utils.R')
source('~/repos/portfolio/demonstrative/R/pbto2/nonlinear_binom_utils.R')

rstan_options(auto_write=T)
options(mc.cores = parallel::detectCores())

static.features <- c('age', 'marshall', 'gcs', 'sex')
ts.features <- c('pbto2', 'paco2')
features <- c(static.features, ts.features)


dsu <- get.long.data(features, scale.vars=F, outcome.func=gos.to.binom, reset.uid=F, rm.na=F)
unscaled.value <- function(x, var) x * sd(dsu[,var], na.rm=T) + mean(dsu[,var], na.rm=T)
d.stan <- dsu %>% mutate_each_(funs(scale(., na.rm=T)), features)

### Stan

d.tr <- d.stan %>% reset.uid
d.ho <- d.stan %>% filter(uid %in% head(unique(d.stan$uid), 2)) %>% reset.uid

na.sentinel <- -999
d.model <- get.stan.data.cv(d.tr, d.ho, static.features, ts.features, n.outcome=2, na.value=na.sentinel)
setwd('~/repos/portfolio/demonstrative/R/pbto2/models/stan')
model.file <- 'nonlinear_binom2_kfold.stan'

posterior <- stan(model.file, data = d.model,
                  warmup = 250, iter = 2500, thin = 5, 
                  chains = 2, verbose = FALSE)

# posterior <- stan(model.file, data = d.model,
#                   warmup = 150, iter = 4000, thin = 5, 
#                   chains = 14, verbose = FALSE)

# Running parallel chains on Mac

library(parallel) # or some other parallelizing package
n.chains <- 5

posterior <- mclapply(1:n.chains, mc.cores = n.chains, FUN = function(chain) {
  stan(file = model.file, data = d.model, warmup = 300, iter = 3000, chains = 1, thin = 5, 
       verbose = FALSE, chain_id=chain)
})
posterior <- sflist2stanfit(posterior)


pars <- c('beta', 'betaz1', 'betaz2', 'a11', 'a21', 'a12', 
          'a22', 'b11', 'b21', 'b12', 'b22', 'c11', 'c21', 'c12', 'c22', 'alpha', 'p1', 'p2')
post <- rstan::extract(posterior)
print(posterior, pars)

rstan::traceplot(posterior, pars)
plot(posterior)


waic <- waic(extract_log_lik(posterior))

get.ts.feature.post.plot <- function(ts.feature, vars){
  x <- seq(min(d.stan[,ts.feature], na.rm=T), max(d.stan[,ts.feature], na.rm=T), length.out = 100)
  get.ts.post.plot(post, vars, x, ts.feature, dsu[,ts.feature])
}
vars1 = c('a1'='a11', 'a2'='a21', 'b1'='b11', 'b2'='b21', 'c1'='c11', 'c2'='c21')
vars2 = c('a1'='a12', 'a2'='a22', 'b1'='b12', 'b2'='b22', 'c1'='c12', 'c2'='c22')
p1 <- get.ts.feature.post.plot(ts.features[1], vars1)
p2 <- get.ts.feature.post.plot(ts.features[2], vars2)


# Static covariate coefficient posterior plot
post.summary <- get.posterior.summary(post, static.features)
p3 <- post.summary %>% 
  filter(!variable %in% c('intercept')) %>%
  ggplot(aes(x=variable, y=mid, ymin=lo, ymax=hi, color=variable)) + 
  geom_pointrange(size=1) + coord_flip() + theme_bw() + 
  geom_hline(yintercept=0, linetype='dashed') + 
  ggtitle('Coefficient 95% Credible Intervals') +
  ylab('Coefficient Range') + xlab('')


# Save above plots to file
filename <- paste(ts.features, collapse='_')
file <- sprintf("~/repos/portfolio/demonstrative/R/pbto2/presentations/images/no_interp/double_var/actual_%s.png", filename) 
png(file = file, width=1200, height=800)
main.title <- sprintf('Model: %s + %s\n(waic = %s +/- %s, pe = %s +/- %s)', 
                      ts.features[1], ts.features[2], round(waic$waic, 3), round(waic$se_waic, 3),
                      round(waic$p_waic, 3), round(waic$se_p_waic, 3))
grid.arrange(p3, arrangeGrob(p1, p2, ncol=2), heights=c(1/3, 2/3), ncol=1, main=main.title)
dev.off()


