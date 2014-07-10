source('tfa.R')

library(gridExtra)
library(ggplot2)
library(reshape2)

# Feature decompositions and plotting functions
tfa_features = new.env()

tfa_features$get_mds = function(data, data_cols=tfa$get_feature_columns()){
  #' Returns the given feature data set scaled to 2 dimensions 
  #' 
  #' Args:
  #'   data: Data set from tfa$get_data()
  #'   data_cols: Columns with data to include in MDS; defaults
  #'      to only columns for feature values
  #' Returns:
  #'   data frame with two numeric columns

  # Restrict to only feature columns (i.e. those not for the track name, artist, or sales)
  feature_data = data[,tfa$get_feature_columns()]
  
  # Scale each value to unit variance and 0 mean
  feature_data = apply(feature_data, 2, scale)
  
  # Calculate distiances and run MDS
  feature_dist = dist(feature_data)
  feature_mds = cmdscale(feature_dist)
  as.data.frame(feature_mds)
}

tfa_features$get_pca = function(data){
  #' Returns PCA object for the given dataset
  #' 
  #' Args:
  #'   data: Data set from tfa$get_data()
  #' Returns:
  #'   list with PCA results
  
  # Restrict to only feature columns (i.e. those not for the track name, artist, or sales)
  feature_data = data[,tfa$get_feature_columns()]
  
  # Scale each value to unit variance and 0 mean
  feature_data = apply(feature_data, 2, scale)
  
  # Return PCA result
  princomp(feature_data)
}


# Load in the raw data set
data = tfa$get_data() 
N = nrow(data)

#############################
# Feature value density plots
#############################

# Stack the data for plotting
stacked = melt(data, id.vars=c('artist_id', 'artist_name', 'track_name', 'sales'))
# > print(head(stacked[sample(1:nrow(stacked), 10),]), row.names=F)
# * Actual sales numbers ommitted for contractual reasons
#  artist_id   artist_name            track_name    sales    variable      value
#     300515     Olly Murs     I Blame Hollywood   xxx.xx         key   4.000000
#     116098    Trey Songz          We Should Be   xxx.xx       tempo 118.341000
#      14249 Janet Jackson 20 Part 4 (Interlude)   xxx.xx speechiness   0.034669

# Plot density by feature
ggplot(stacked, aes(x=value)) +  
  geom_histogram(aes(y=..density..), fill=NA, color='black') + 
  geom_density(color='blue') + 
  facet_wrap(~ variable, scales="free") +
  ggtitle('Track Feature Densities') + 
  xlab('Feature Value') + ylab('Density')


##################################
# Feature value scaling/clustering
##################################

# Sample the raw data before running MDS to avoid performance issues
set.seed(123)
data_sample = data[sample(1:nrow(data), 10000),]

# Run MDS and return 2-D data frame
feature_mds = tfa_features$get_mds(data_sample)

# Create scatterplot and densities for results
g1 = ggplot(feature_mds, aes(x=V1, y=V2)) + 
  geom_point() + ggtitle('2-D Feature Scaling Scatterplot')
g2 = ggplot(feature_mds, aes(x=V1, y=V2)) + 
  stat_density2d(aes(alpha=..level.., fill=..level.., color=..level..), geom="polygon") + 
  ggtitle('2-D Feature Scaling Density') + theme(legend.position="none")
do.call(grid.arrange, list(g1, g2, ncol=2))



##############################
# Feature principal components
##############################

# Get row id for extreme outlier 'Ik Onkar' by AR Rahman
ik_onkar_row = as.integer(rownames(subset(data, artist_id == 65923 & track_name == 'Ik Onkar'))[1])
par(mfrow=c(1,2))

# Create biplot for principal components within entire data set (and label outlier)
feature_pca = tfa_features$get_pca(data)
labs = as.character(ifelse(1:N == ik_onkar_row, '. "Ik Onkar" by AR Rahman', rep(".", N)))
biplot(feature_pca, xlabs=labs, cex=1, col=c('#999999', 'red'), expand=.8, main='Feature Bi-Plot')

# Create biplot with outlier removed
feature_pca = tfa_features$get_pca(data[-ik_onkar_row,])
biplot(feature_pca, xlabs=rep(".", N-1), cex=1, col=c('#999999', 'red'), expand=.8, main='Feature Bi-Plot (w/o "Ik Onkar")')

