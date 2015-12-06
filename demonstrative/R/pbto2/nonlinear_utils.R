
get.stan.data <- function(d.stan, static.features, ts.feature){
  d.stan <- data.frame(d.stan)
  d.stan.uid <- d.stan %>% group_by(uid) %>% do({head(., 1)}) %>% ungroup %>% arrange(uid) %>% data.frame
  list(
    N_OBS = nrow(d.stan),
    N_VARS = length(static.features),
    N_UID = max(d.stan$uid),
    y = d.stan.uid %>% .$outcome %>% as.integer,
    x = d.stan.uid[,static.features],
    z = d.stan[,ts.feature],
    uid = d.stan$uid
  )
}

double.logistic <- function(x, a1, a2, b1, b2, c1, c2, c=0){
  r1 <- a1 / (1 + exp(b1 * (x - c1)))
  r2 <- a2 / (1 + exp(b2 * (x - c2)))
  #.5 * (r1 + r2)
  c + r1 + r2
}

get.mean.curve <- function(post, x){
  a1 = mean(post$a1)
  a2 = mean(post$a2)
  b1 = mean(post$b1)
  b2 = mean(post$b2)
  c1 = mean(post$c[,1])
  c2 = mean(post$c[,2])
  double.logistic(x, a1, a2, b1, b2, c1, c2)
}

