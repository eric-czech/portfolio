library(foreach)
library(gbm)
library(dplyr)
library(stringr)
library(caret)
library(doParallel)
registerDoParallel(makeCluster(5))

setwd('/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/')
source('utils.R')

data <- get.raw.data()
data <- data %>% mutate(Country = factor(Country))
head(data)

options(warn=0)
n.bootstrap <- 10

gbrt.predictions <- foreach(i=1:n.bootstrap, .combine=rbind) %do% {
  samp <- data %>% sample_frac(size=1, replace=T)
  fit <- gbm( Homicide.Rate ~ Year + Country, data = samp, n.trees=100, distribution='gaussian')
  data.frame(Country=unique(samp$Country), Year=2015-2000) %>%
    mutate(Prediction=predict(fit, ., n.trees=100), Year=Year+2000, SampleId=i)
}

gbrt.predictions <- foreach(i=1:n.bootstrap, .combine=rbind) %dopar% {
  library(dplyr)
  library(caret)

  samp <- data %>% sample_frac(size=1, replace=T)

  tuneGrid <- expand.grid(
    sigma = c(1, 3, 7), 
    n.trees = c(100, 500, 1000), 
    shrinkage = c(.5, .1), 
    n.minobsinnode = 1
  )
  fit <- train(Homicide.Rate ~ Year + Country, data=data, method="svmRadial", trControl=trainControl(method='cv', number=10))
  fit <- train(
    Homicide.Rate ~ Year + Country, data = samp, tuneGrid = tuneGrid, method = "gbm", 
    trControl = trainControl(method = "cv", number = 10, allowParallel = F), verbose = FALSE, distribution='laplace' 
  )  
  
  data.frame(Country=unique(samp$Country), Year=2015-2000) %>%
    mutate(Prediction=predict(fit, .), Year=Year+2000, SampleId=i)
}



library(randomForest)
fit <- gbm(Homicide.Rate ~ Year + Country, data = d, interaction.depth = 17, n.trees = 100, shrinkage = .5, var.monotone = c(1, 0))
#fit <- randomForest(Homicide.Rate ~ Year + Country, data = data)
#predict(fit, data.frame(Country='Saint Kitts and Nevis', Year=2015 - 2000), n.trees=100)
fit <- train(Homicide.Rate ~ Year + Country, data=data, method="svmRadial", trControl=trainControl(method='cv', number=10))
p <- data.frame(Country=data$Country[data$Country == 'Saint Kitts and Nevis'][1], Year=2000:2015 - 2000)
predict(fit, newdata=p)

ordered.countries <- gbrt.predictions %>% group_by(Country) %>% 
  dplyr::summarise(Median=median(Prediction)) %>%
  arrange(desc(Median)) %>% .$Country

gbrt.predictions %>% 
  mutate(Country=factor(as.character(Country), , levels=ordered.countries)) %>%
  ggplot(aes(x=Country, y=Prediction)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#varImp(fit$finalModel) %>% add_rownames(var='Feature') %>% ggplot(aes(x=Feature, y=Overall)) + geom_bar(stat='identity')