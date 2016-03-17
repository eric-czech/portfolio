
source('/Users/eczech/repos/misc/bemkl/bemkl_supervised_multilabel_classification_variational_train.R')
source('/Users/eczech/repos/misc/bemkl/bemkl_supervised_multilabel_classification_variational_test.R')

##### Generate Data #####
library(plyr)
library(dplyr)
library(caret)
library(kernlab)
library(abind)

C <- 4
N <- C * 1000
X <- matrix(rnorm(n=N), nrow=N/C)
create.res <- function(r, noise.sd=.01) {
  p <- 1 / (1 + exp(-sum(r) + rnorm(1, 0, noise.sd)))
  sample(c(1, -1), size = 1, prob = c(p, 1-p))
}
X1 <- X[,1:floor(C/2)]
X2 <- X[,(floor(C/2)+1):C]

Y <- apply(X, 1, function(r){
  c(
    y1=create.res(r[c(1)]),
    y2=create.res(r[c(1)])
  )
})
Y <- t(Y)

get.kernel.matrix <- function(X){
  kf <- polydot(offset=0)
  as.matrix(kernelMatrix(kf, X))  
}

kX1 <- get.kernel.matrix(X1)
kX2 <- get.kernel.matrix(X2)
kX <- do.call(abind, list(list(kX1, kX2), along=3))

boxplot(X1[,1] ~ Y[,1])
##### Run Model #####

#initalize the parameters of the algorithm
parameters <- list()

#set the hyperparameters of gamma prior used for sample weights
parameters$alpha_lambda <- 1
parameters$beta_lambda <- 1

#set the hyperparameters of gamma prior used for bias
parameters$alpha_gamma <- 1
parameters$beta_gamma <- 1

#set the hyperparameters of gamma prior used for kernel weights
parameters$alpha_omega <- 1
parameters$beta_omega <- 1

### IMPORTANT ###
#For gamma priors, you can experiment with three different (alpha, beta) values
#(1, 1) => default priors
#(1e-10, 1e+10) => good for obtaining sparsity
#(1e-10, 1e-10) => good for small sample size problems

#set the number of iterations
parameters$iteration <- 200

#set the margin parameter
parameters$margin <- 1

#determine whether you want to store the lower bound values
parameters$progress <- 0

#set the seed for random number generator used to initalize random variables
parameters$seed <- 1606

#set the standard deviation of intermediate representations
parameters$sigma_g <- 0.1

#set the number of labels
L <- 2
#set the number of kernels
P <- 2

#initialize the kernels and class labels for training
Ktrain <- kX #should be an Ntra x Ntra x P matrix containing similarity values between training samples
Ytrain <- t(Y) #should be an L x Ntra matrix containing class labels (contains only -1s and +1s) where L is the number of labels

#perform training
state <- bemkl_supervised_multilabel_classification_variational_train(Ktrain, Ytrain, parameters)

#display the kernel weights
print(state$be$mu[(L + 1):(L + P)])

#initialize the kernels for testing
Ktest <- ?? #should be an Ntra x Ntest x P matrix containing similarity values between training and test samples

#perform prediction
prediction <- bemkl_supervised_multilabel_classification_variational_test(Ktest, state)

#display the predicted probabilities
print(prediction$P)