library(foreach)
library(gbm)
library(dplyr)
library(stringr)
library(caret)
library(doParallel)
library(kernlab)
registerDoParallel(makeCluster(5))

setwd('/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/')
source('utils.R')

data <- get.raw.data()
data <- data %>% mutate(Country = factor(Country))
head(data)

options(warn=0)
n.bootstrap <- 30

svm.predictions <- foreach(i=1:n.bootstrap, .combine=rbind) %do% {
  samp <- data %>% sample_frac(size=1, replace=T)
  fit <- ksvm(Homicide.Rate ~ Year + Country, data = samp, kernel='rbfdot')
  data.frame(Country=unique(samp$Country), Year=2015-2000) %>%
    mutate(Prediction=predict(fit, .)[,1], Year=Year+2000, SampleId=i)
}


svm.predictions <- foreach(i=1:n.bootstrap, .combine=rbind) %dopar% {
  library(dplyr)
  library(caret)
  
  samp <- data %>% sample_frac(size=1, replace=T)
  
  tuneGrid <- expand.grid(C=c(.001,.01,.1,.5,1), degree=c(2), scale=c(.05,.1))
  fit <- train(Homicide.Rate ~ Year + Country, data=data, method="svmPoly", 
               preProc = c("center", "scale"), tuneGrid = tuneGrid,
               trControl=trainControl(method='cv', number=10, allowParallel = F))
  
  data.frame(Country=unique(samp$Country), Year=2015-2000) %>%
    mutate(Prediction=predict(fit, .), Year=Year+2000, SampleId=i)
}

ordered.countries <- svm.predictions %>% group_by(Country) %>% 
  dplyr::summarise(Median=median(Prediction)) %>%
  arrange(desc(Median)) %>% .$Country

svm.predictions %>% 
  mutate(Country=factor(as.character(Country), , levels=ordered.countries)) %>%
  ggplot(aes(x=Country, y=Prediction)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



# Tests

i = pi/8
x = seq(-4*pi, 4*pi, i)
xn = x
#d = data.frame(x=x, y=sin(x))
d = data.frame(x=x, y=sin(x)/x + .01 * x, Type='Actual')
plot(d$x, d$y)

tuneGrid <- expand.grid(C=c(.1,1,10,100), sigma=c(.1,1,10,100))
fit <- train(y ~ x, data=d, method="svmRadial", tuneGrid=tuneGrid,
             preProc = c("center", "scale"), trControl=trainControl(method='cv', number=10, allowParallel = T))
print(fit$bestTune)
y = predict(fit, newdata=data.frame(x=xn))
d = rbind(d, data.frame(x=xn, y=y, Type='Predicted'))
ggplot() + 
  geom_line(data=subset(d, Type=='Actual'), aes(x=x, y=y)) + 
  geom_point(data=subset(d, Type=='Predicted'), aes(x=x, y=y))


