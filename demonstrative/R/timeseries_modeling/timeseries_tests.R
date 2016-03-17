library(dplyr)
library(plyr)
library(caret)
library(ggplot2)

dt <- ozone[1,1,]
x1 <- seq_along(dt)
x2 <- x1#x1[2:length(x1)]
X1 <- data.frame(x=x1, p=x1 %% 12)
X2 <- data.frame(x=x2, p=x2 %% 12)
y1 <- dt
y2 <- as.numeric(scale(dt))#as.numeric(scale(diff(dt)))

m <- gausspr(X2, y2, kpar=list(sigma=.3))

#m <- train(Xt, y, method='gaussprRadial', trControl=trainControl(method='none'))
#m <- train(Xt, y, method='glm', trControl=trainControl(method='none'))
#m <- train(Xt, y, method='gbm', trControl=trainControl(method='cv', number=10), verbose=F)
#m <- gausspr(Xt, y)
#m <- train(Xt, y, method='gam', tuneLength=10, trControl=trainControl(method='cv'))

#X1n <- data.frame(x=c(X1$x, max(X1$x) + 1:50)) %>% mutate(p=x%%12)
X2n <- data.frame(x=c(X2$x, max(X2$x) + 1:50)) %>% mutate(p=x%%12)

yn <- predict(m, X2n)[,1]
d <- rbind(
  X2 %>% mutate(y=y2, type='o'),
  X2n %>% mutate(y=yn, type='p')
)
d %>% ggplot(aes(x=x, y=y, color=type)) + geom_line()