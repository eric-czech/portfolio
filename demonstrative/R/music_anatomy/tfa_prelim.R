source('tfa.R')

tfa_features = new.env()
data = tfa$get_data()

loginfo('Raw data sample:')
print(head(data[sample(1:nrow(data), 10),]), row.names=F)

loginfo('Total number of records in data set: ')
nrow(data)

loginfo('Number of distinct artist and track names in data set: ')
apply(data[,c('artist_name', 'track_name')], 2, function(x){length(unique(x))})
