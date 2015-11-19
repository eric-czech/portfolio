
d <- read.csv('/Users/eczech/data/ptbo2/export/data_wide_bin_48hr.csv')
names(d)

features <- c('age', 'marshall', 'gcs', 'sex')

scale <- function(x) (x - mean(x)) / sd(x)
d.m <- d %>% 
  rename(outcome=gos.3.favorable) %>% 
  mutate_each_(funs(scale), features) %>%
  dplyr::select(starts_with('pbto2_ct_bin'), one_of(features), outcome)

m2 <- glm(outcome ~ ., data=d.m, family='binomial')