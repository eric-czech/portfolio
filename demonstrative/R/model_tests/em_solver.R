

# Create sample by combining two samples from different poisson distributions
s1 = rpois(n = 500, lambda=100)
s2 = rpois(n = 500, lambda=1)
x <- sample(c(s1, s2), replace=F)

# Initialize lambda parameter guesses
#theta <- runif(2, .1, 250)
theta <- c(150, 200)

# Set max iterations
max.i <- 50
i <<- 1

# Optimize
repeat {
  i <<- i + 1
  
  print(sprintf('Begging step %s with theta = %s', i, paste(theta, collapse=', ')))
  
  # E step --> p(Z|X;T0)
  Z <- data.frame(c1=dpois(x, theta[1]), c2=dpois(x, theta[2]))
  Z <- Z / apply(Z, 1, sum)
  
  lik <- Z[,1] * dpois(x, theta[1])
  if (any(is.na(Z))){
    browser()
    stop('Found nas')
  }
  
  # M step argmax{T} --> p(Z|X;T0) ln(p(X, Z|T))
  obj.fun <- function(p){
    v <- sapply(seq_along(x), function(i){
      p1 <- max(log(dpois(x[i], lambda=p[1])), -99999)
      p2 <- max(log(dpois(x[i], lambda=p[2])), -99999)
      Z[i,1] * p1 + Z[i,2] * p2
    })
    sum(v)
  }
  res <- optim(theta, obj.fun, method='L-BFGS-B', lower=c(.1, .1), upper=c(1000, 1000), control=list(fnscale=-1))
  
  print(sprintf('Convergence = %s, Value = %s, Parameters = (%s, %s)', res$convergence, res$value, res$par[1], res$par[2]))

  theta <<- res$par
  if (i == max.i)
    break
} 