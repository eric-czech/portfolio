library(plyr)
library(dplyr)
library(caret)
library(caretEnsemble)
library(devtools)
#devtools::load_all('/Users/eczech/repos/misc/caretEnsemble')
#devtools::test('/Users/eczech/repos/misc/caretEnsemble')

d <- twoClassSim(n=100)
X <- d %>% select(-Class)
y <- d$Class

customRF <- getModelInfo('rf', regex=F)[[1]]
customRF$method <- 'custom.rf'

customGLM <- getModelInfo('rf', regex=F)[[1]]
customGLM$method <- 'custom.glm'

cl <- caretList(
  X, y,
  tuneList=list(
    # The name used internally for this model will come from the "method" attribute above
    caretModelSpec(method=customRF, tuneLength=3),
    # This model on the other hand will be referred to as "myglm" not "custom.glm"
    myglm=caretModelSpec(method=customGLM, tuneLength=1),
    glmnet=caretModelSpec(method='glmnet', tuneLength=5),
    rpart=caretModelSpec(method='rpart', tuneLength=15)
  ),
  trControl=trainControl(method='cv', number=10, classProbs=T)
)

cs <- caretEnsemble(cl)

cs