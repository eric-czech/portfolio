source('tfa.R')

library(ggplot2)
library(reshape2)
library(scales)

data = tfa$get_data() 

# Look at top-level numbers for sales
summary(data$sales)

# They're clearly not from a symmetric distribution,
# so lets so how bad the skew is
data.frame(value=quantile(data$sales, probs = seq(0, 1, .1)))

# Plot sales on original and log scale
g = ggplot(data, aes(x=sales)) + geom_freqpoly() + xlab('Sales Value') + ylab('Frequency')

g1 = g + ggtitle('Frequency of Sales Numbers')
g2 = g + ggtitle('Frequency of Log Sales Numbers') + scale_x_log10(labels=comma)
  
do.call(grid.arrange, list(g1, g2, ncol=2))
