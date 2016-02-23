
scale.df <- function(d) d %>% mutate_each(funs(scale), -gos, -uid)

prep.df <- function(d, ts.features, scale.vars=T){
  if (length(ts.features) > 0){
    ts.na <- d %>% select_(.dots=paste0(ts.features, '_is_na')) %>% apply(1, sum)
    d <- d %>% filter(ts.na == 0)
  }
  if (scale.vars) {
    d %>% select(-ends_with('_is_na')) %>% scale.df
  } else {
    d %>% select(-ends_with('_is_na'))
  }
}