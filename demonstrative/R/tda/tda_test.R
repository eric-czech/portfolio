d.file <- '/Users/eczech/data/meetups/genomics/data_subset_v2.csv'
d <- read.csv(d.file, sep=',')

d.dist <- dist(d)

library(TDAmapper)
library(fastcluster) 
library(foreach)
library(iterators)

eps <- 1000
d.mat <- as.matrix(d)
filter_values <- foreach(i=1:nrow(d.mat)) %do% {
  ri <- d.mat[i,]
  y <- foreach(j=1:nrow(d.mat), combine=c) %do% {
#     if (j == i)
#       return(NULL)
    dt <- ri - d.mat[j,]
    dt <- (dt %*% dt)[1,1]
    exp(-dt/eps)
  }
  sum(unlist(y))
}

length(filter_values)


m1 <- mapper1D(
  distance_matrix = d.dist, 
  filter_values = unlist(filter_values),
  num_intervals = 10,
  percent_overlap = 50,
  num_bins_when_clustering = 10)

install.packages("igraph") 
library(igraph)

g1 <- graph.adjacency(m1$adjacency, mode="undirected")
plot(g1, layout = layout.auto(g1) )
tkplot(g1)

library(networkD3)
MapperNodes <- mapperVertices(m1, nrow(d) )
MapperLinks <- mapperEdges(m5)



