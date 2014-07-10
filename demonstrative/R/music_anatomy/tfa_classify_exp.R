source('tfa.R')
source('model_utils.R')

library(plyr)
library(gbm)

# Classification experiments and tests

tfa_classify_exp = new.env()

data = tfa$get_data() 

#' Same as tfa_classify$get_modeling_results but uses cforest instead of randomForest
tfa_classify_exp$get_modeling_results_cf = function(data, k){
  trainer = function(data){
    n = names(data)
    data = data[,n[n != 'sales']]
    res = cforest(is_hit ~ ., data = data) 
    res
  }
  predictor = function(fit, data){
    ldply(predict(fit, newdata=data, type='prob'), function(x){x[2]})$V1
  }
  fetch_var_imp = function(fit){
    varimp(fit)
  }
  fetch_classes = function(fit, data){
    predict(fit, newdata=data)
  }  
  model_utils$cross_validate(
    data, trainer, predictor, 'is_hit', k=k, 
    .fetch_var_imp=fetch_var_imp, .fetch_classes=fetch_classes, 
    .smote_training=F, .type='interleaved'
  )
}

#' Same as tfa_classify$get_modeling_results but uses gbm instead of randomForest
tfa_classify_exp$get_modeling_results_gbm = function(data, k){
  trainer = function(data){
    data = transform(data, is_hit = as.numeric(as.character(is_hit)))
    gbm(is_hit ~ . - sales, data = data, 
        distribution = "bernoulli", 
        shrinkage = .001, 
        n.trees = 3000, 
        interaction.depth = 2)
  }
  predictor = function(fit, data){
    ntrees = gbm.perf(fit, method = "OOB", plot.it=F)
    predict(fit, newdata = data, n.trees=ntrees, type="response")
  }
  fetch_var_imp = function(fit){
    ntrees = gbm.perf(fit, method = "OOB", plot.it=F)
    var_importance = relative.influence(fit, n.trees=ntrees)
  }
  fetch_classes = function(fit, data){
    factor(ifelse(predictor(fit, data) > .5, 1, 0), levels=c(0,1))
  }  
  model_utils$cross_validate(
    data, trainer, predictor, 'is_hit', k=k, 
    .fetch_var_imp=fetch_var_imp, .fetch_classes=fetch_classes, 
    .smote_training=F, .type='interleaved'
  )
}

#########################
# Modeling with artist_id
#########################

# Determine number of tracks per artist
counts = ddply(data, .(artist_id), transform, num_tracks=length(unique(track_name)))

# Minimum number of tracks to allow
min_tracks = 25

# Restrict to artists with required number of tracks
counts = subset(counts, num_tracks >= min_tracks)
nrow(counts)
# 32310 - 87% of total

# Get unique artist set
applicable_artists = unique(subset(counts, num_tracks >= min_tracks)$artist_id)

# Fetch raw data (with artist_id) and create 'is_hit' variable
model_data = data[,c('sales', tfa$get_feature_columns(), 'artist_id')]
model_data = subset(model_data, artist_id %in% applicable_artists)
results_w_artist = tfa_classify$run_model(model_data)
tfa_classify$plot_modeling_results(results_w_artist, k)
