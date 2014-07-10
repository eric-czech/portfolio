source('tfa.R')
source('model_utils.R')

library(randomForest)
library(ROCR)
library(plyr)

# Classification functions and analysis
tfa_classify = new.env()

data = tfa$get_data() 

tfa_classify$get_modeling_results = function(data, k){
  #' Runs random forest model for k folds of the given input data set
  #' and returns model fit, predicted probabilities, predicted classes,
  #' and variable importance for each fold

  trainer = function(data){
    randomForest(is_hit ~ . - sales, data, family='binomial', ntree=250)
  }
  predictor = function(fit, data){
    predict(fit, data, type='prob')[,2]
  }
  fetch_var_imp = function(fit){
    importance(fit)[,1]
  }
  fetch_classes = function(fit, data){
    predict(fit, data, type='response')
  }  
  model_utils$cross_validate(
    data, trainer, predictor, 'is_hit', k=k, 
    .fetch_var_imp=fetch_var_imp, .fetch_classes=fetch_classes, 
    .smote_training=F, .type='interleaved'
  )
}


tfa_classify$get_perf_summary = function(results, k){
  #' Returns performance measures for the model results including
  #' True/False Positive/Negative counts
  
  performance = foreach(i = 1:k, .combine=rbind) %do% {
    actual = results$response[[i]]
    predicted = results$classes[[i]]    
    conf_matrix = data.frame(table(actual, predicted))
    names(conf_matrix) = c('actual', 'predicted', 'count')
    conf_matrix$percentage = 100 * conf_matrix$count / length(actual)
    cbind(fold=i, conf_matrix)
  }
  
  labeller = function(data){
    act = data['actual']
    pre = data['predicted']
    if (act == pre && pre == '1')
      return('True Positive')
    if (act != pre && pre == '1')
      return('False Positive')
    if (act == pre && pre == '0')
      return('True Negative')
    if (act != pre && pre == '0')
      return('False Negative')
  }
  performance$type = apply(performance, MARGIN=1, FUN=labeller)
  transform(performance, type = factor(type), fold = factor(fold))
}


tfa_classify$plot_modeling_results = function(results, k){
  #' Plots the following performance measures:
  #' 1. Accuracy vs Probability cutoff (via ROCR)
  #' 2. Boxplot of True/False Positive/Negative counts for each fold
  #' 3. Boxplot of variable importance for each fold
  
  tryCatch({dev.off()})
  par(mfrow=c(1,3))
  
  # Plot accuracy vs probabilities
  pred = prediction(results$predictions, results$response)
  perf = performance(pred, 'acc')
  plot(perf, main="Accuracy vs. Probability")
  
  # Plot overall performance
  boxplot(percentage ~ type, data = tfa_classify$get_perf_summary(results, k), main="Performance", ylab="Percentage", xlab="Prediction Type")
  
  # Plot variable importance
  var_imp = ldply(results$var_imp, unlist)
  var_names = names(sort(apply(var_imp, MARGIN=2, mean), decreasing=T))
  var_imp = var_imp[,var_names]
  boxplot(var_imp, las=2, main="Variable Importance", xlab="", xaxt = "n")
  text(x =  seq_along(var_names), y = par("usr")[3] - 1, srt = 45, adj = 1.1,
       labels = var_names, xpd = TRUE)
}


#################
# Modeling basics
#################
k = 10
percentile_cutoff = .5

tfa_classify$add_response = function(data){
  # Determine sales cutoff at percentile and use it to create the 'is_hit' variable to be predicted
  sales_at_cutoff = quantile(data$sales, probs=percentile_cutoff)
  data$is_hit = factor(as.numeric(data$sales > sales_at_cutoff))
  data
}

tfa_classify$run_model = function(data){
  data = tfa_classify$add_response(data)
  
  # Run random forest and plot performance measures
  tfa_classify$get_modeling_results(data, k)
}

#############################
# Modeling with features only
#############################

# Fetch raw data with no artist_id and run modeller
model_data = data[,c('sales', tfa$get_feature_columns())]
results_feature_only = tfa_classify$run_model(model_data)
tfa_classify$plot_modeling_results(results_feature_only, k)

###################################
# Modeling with track count as well
###################################

# Fetch raw data, add track count as a feature, and run modeller
extra_cols = c('artist_id', 'track_name')
model_data = data[,c('sales', tfa$get_feature_columns(), extra_cols)]
model_data = ddply(model_data, .(artist_id), transform, num_tracks=length(unique(track_name)))
model_data = model_data[, names(model_data)[!names(model_data) %in% extra_cols]]
results_w_track_count = tfa_classify$run_model(model_data)
tfa_classify$plot_modeling_results(results_w_track_count, k)

##########################
# Model comparison testing
##########################
matches = data.frame()
for (i in 1:k){
  r1 = results_feature_only$response[[i]] == results_feature_only$classes[[i]]
  r2 = results_w_track_count$response[[i]] == results_w_track_count$classes[[i]]
  matches = rbind(matches, data.frame(r1, r2))
  conf_matrix = table(r1, r2)
  # Print mcnemar results for each fold
  print(mcnemar.test(conf_matrix))
}
# Also print mcnemar results for pooled results
mcnemar.test(table(matches$r1, matches$r2))


################
# Decision trees
################
library(rpart)
library(rpart.plot)
library(rattle)

# Creating training and testing sets
dt_data = tfa_classify$add_response(model_data)
n = names(dt_data)
dt_data = dt_data[,n[n != 'sales']]
i = sample(1:nrow(dt_data), .75 * nrow(dt_data))
training = dt_data[i,]
testing = dt_data[-i,]

# Run decision tree via rpart
rp = rpart(is_hit ~ ., data = training, control=rpart.control(cp=.01)) 

# Check performance -- non-ensemble trees don't generalize as well but still run at around 60% accuracy
predicted_classes = predict(rp, newdata = testing, type='class')
table(predicted_classes, testing$is_hit)

# Plot the decision tree
fancyRpartPlot(rp)

# Show examples from each split in the tree
examples = data[i,]
examples$predicted = predict(rp, newdata = training, type='prob')[,2]
examples = examples[,c('artist_id', 'artist_name', 'track_name', 'predicted', 'sales', 'duration', 'loudness', 'acousticness', 'valence')]

head(arrange(subset(examples, duration < 172.9862) , desc(predicted)), 10)
head(arrange(subset(examples, duration >= 172.9862 & loudness >= -5.9795) , desc(predicted)), 10)
head(arrange(subset(examples, duration >= 172.9862 & loudness < -5.9795 & acousticness >= 0.5850815) , desc(predicted)), 10)
head(arrange(subset(examples, duration >= 172.9862 & loudness < -5.9795 & acousticness < 0.5850815 & valence >= 0.5698035) , desc(predicted)), 10)
head(arrange(subset(examples, duration >= 172.9862 & loudness < -5.9795 & acousticness < 0.5850815 & valence < 0.5698035) , desc(predicted)), 10)

