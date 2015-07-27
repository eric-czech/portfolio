library(dplyr)
library(glmnet)

d <- read.table('/tmp/MeetupDataChallengeFlat.csv', sep=',', header = T)
d %>% sapply(class)

id.cols = c('Run.Number', 'Yield', 'Purity')
num.cols = names(d)[!names(d) %in% id.cols]

convert.col <- function(x){
  if (length(unique(x)) == 1)
    x
  else
    factor(x)
}
d[,num.cols] <- d[,num.cols] %>% mutate_each(funs(convert.col))

form.1 <- paste(num.cols, collapse=' + ')
form.1 <- paste('Yield ~ ', form.1)

X <- model.matrix(as.formula(form.1), data=d)

fit <- cv.glmnet(X, d[,'Yield'], family='gaussian')

