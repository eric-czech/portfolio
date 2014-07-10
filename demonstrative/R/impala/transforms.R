source('~/NBS-DS/R/impala/meta.R')

library(data.table)
library(zoo)

#' Common transformations meant to be used in conjunction with raw Impala results.
#' 
#' > Unit tests in transforms_utest.R
#' > Performance tests in transforms_ptest.R
#' 
#' Example:
#' 
#' 1. "UnstackTimeseries"
#' 
#' > metric_data = impalaQuery(query, col.names=COLS_IDX_ENTITY)
#' account_id entity_id endpoint_id                 endpoint_identifier metric_id count_type unix_seconds value
#' 1         NA      9204     8714760 virtuallabel-6OhVFDB142MxB2VGB8iG34       258          d   1398729600     1
#' 2         NA      9206      184645                       leaving+araby         8          t   1396051200   685
#' 3         NA      9207      184646                             shurman         9          t   1399507200  2188
#'
#' > unstacked = UnstackTimeseries(metric_data)
#' # Show FB Page Likes for OneRepublic
#' > arrange(subset(unstacked, entity_id == 63 & metric_id == 11), unix_seconds)
#'      entity_id account_id metric_id unix_seconds delta_value total_value is_interpolated
#'   1:        63         NA        11   1388534400       12198     5329103           FALSE
#'   2:        63         NA        11   1388620800       15102     5341301           FALSE
#'  ---                                                                                    
#' 145:        63         NA        11   1400976000       14946     9259654           FALSE
#' 146:        63         NA        11   1401062400          NA     9274600           FALSE



 
UnstackTimeseries = function(metric_data, dimensions = c()){
  #' "Unstacks" metric timeseries data by returning a result with
  #' a total value AND a delta value for each data point.  
  #' 
  #' Input data frames are expected in the form:
  #'    { entity_id, account_id, count_type, metric_id, unix_seconds, value [, dimensions ] }
  #'    - This is the same structure as the results of raw impala queries
  #'    - The optional 'dimensions' only apply when querying for data at a higher granularity than global
  #' 
  #' There are 3 primary ways that data is "unstacked":
  #' 
  #' 1. For metrics that only have total values, a delta value is 
  #' computed after interpolating the given total values.  In this
  #' case, the 'is_interpolated' column in the result indicates
  #' whether or not the *total* value for that particular date
  #' was estimated (linearly) rather than being originally present.
  #' 
  #' 2. For metrics that only have delta values, the total values
  #' returned will be NA and the delta values will be returned as is.
  #' 
  #' 3. For metrics that are pre-aggregated, the total values are 
  #' returned as is and the delta values are similarly unchanged except
  #' for the very last delta value.  It is set as NA to maintain consistency
  #' with delta values calculated for metrics in case #1
  #' 
  #' Note that the total date span of each timeseries will be equal to
  #' the min and max present for each metric.  For total values, the results
  #' are inclusive on that range while delta values are EXCLUSIVE on that range.
  #' This helps make sure the that subtraction of totals is consistent with 
  #' the sum of deltas (where applicable)
  #' 
  #' Args:
  #'  metric_data: data frame returned from impala query
  #'  dimensions: list of dimension names also present in 'metric_data' 
  #'    (e.g. 'country', 'region', 'demographic', etc.)
  #'    
  #' Returns:
  #'  data.table with form { entity_id, account_id, metric_id, unix_seconds, delta_value, total_value, is_interpolated [, dimensions ] }
  #'  Example:
  #'  > UnstackTimeseries(impalaQuery(query, col.names=COLS_IDX_ENTITY))
  #'  # OneRepublic Twitter Follower Data -->
  #'      entity_id account_id metric_id unix_seconds delta_value total_value is_interpolated
  #'   1:        63         NA        11   1388534400       12198     5329103           FALSE
  #'   2:        63         NA        11   1388620800       15102     5341301           FALSE
  #'  ---                                                                                    
  #' 145:        63         NA        11   1400976000       14946     9259654           FALSE
  #' 146:        63         NA        11   1401062400          NA     9274600           FALSE
  #'
    
  required_fields = c('entity_id', 'account_id', 'count_type', 'metric_id', 'unix_seconds', 'value', dimensions)
  n = names(metric_data)
  if (!all(required_fields %in% n))    
    stop(paste('Failed to find at lest on required field in input frame.  
               Fields required: ', paste(required_fields, collapse=", "),'  
               Fields given: ', paste(n, collapse=", ")))
  
  key_fields = c('entity_id', 'account_id', 'metric_id', dimensions)
  metric_data = data.table(metric_data, key = key_fields)
  
  metric_data[,.UnstackTimeseries(metric_id, count_type, unix_seconds, value),by=key_fields]
}

.UnstackTimeseries = function(metric_id, count_type, unix_seconds, value){
  #' Used on a per entity + account + metric basis to "unstack" timeseries into single
  #' data points, each having a total and delta value
  
  if (length(metric_id) == 0)
    return(NULL)
  
  # Determine whether or not this metric is 'pre-aggregated' (i.e. we 
  # can expect deltas AND totals to already be present)
  is_pre_aggregated = metric_id[1] %in% .GetAggregatedMetrics()[,'id']
  
  selector = count_type == 't'
  # If there are total values, extract them into a separate data.table
  if (any(selector)){
    total_values = data.table(
      unix_seconds = unix_seconds[selector],
      total_value = as.numeric(value[selector]),
      is_interpolated = FALSE,
      key = 'unix_seconds'
    )
    
    # Otherwise, create a single row data.table with default values.  
    # This is necessary because data.tables cannot have empty columns and 
    # at least one row must be present for the later join of deltas + totals
  } else {
    total_values = data.table(
      unix_seconds = as.integer(NA), total_value = as.numeric(NA), is_interpolated = FALSE, key = 'unix_seconds'
    )
  }
  
  selector = count_type == 'd'
  # If there are delta values, extract them into a separate data.table
  if (any(selector)){
    delta_values = data.table(
      unix_seconds = unix_seconds[selector],
      delta_value = as.numeric(value[selector]),
      key = 'unix_seconds'
    )    
    delta_values$delta_value[length(delta_values$delta_value)] = as.numeric(NA)
    
  # Otherwise, create a single row data.table with default values.  
  } else {
    delta_values = data.table(
      unix_seconds = as.integer(NA), delta_value = as.numeric(NA), key = 'unix_seconds'
    )
  }
  
  # If we're not dealing with a pre-aggregated metric
  # and we have total values, compute a delta value as well.
  if (!is_pre_aggregated && 't' %in% count_type){  
    # First interpolate the totals
    total_values = .InterpolateTimeseries(total_values$unix_seconds, total_values$total_value)
    
    # Now difference those in-order totals to form deltas
    delta_values = data.table(
      unix_seconds = total_values$unix_seconds,
      delta_value = c(diff(total_values$total_value), NA),
      key = 'unix_seconds'
    )
  }
  
  # Join deltas and totals being careful to make sure that the join doesn't lose values for one 
  # or the other.  Joins with data.tables are 'right outer' by default so starting that chain
  # of joins here with the full unique set of timestamps ensures no data points are lost
  # see: http://stackoverflow.com/questions/12773822/why-does-xy-join-of-data-tables-not-allow-a-full-outer-join-or-a-left-join
  all_unix_seconds = unique(c(total_values$unix_seconds, delta_values$unix_seconds))
  all_unix_seconds = as.integer(all_unix_seconds[!is.na(all_unix_seconds)])
  return(delta_values[total_values[J(all_unix_seconds)]])
  
}



.InterpolateTimeseries = function(unix_seconds, value){
  #' Interpolates missing values using the na.approx function (from zoo)
  #' 
  #' The values given are interpolated between and 
  #' including the first and last date in the data
  
  if (length(unix_seconds) != length(value))
    stop('Timeseries for interpolation cannot have a different number of values and timestamps')
  
  # Initialize timeseries with is interpolated flag set to false
  data = data.table(
    unix_seconds=unix_seconds, 
    total_value=value, 
    is_interpolated=rep(F, length(unix_seconds)),
    key = 'unix_seconds'
  )
  
  # Sanity checks to short-circuit execution and return default results
  if (length(value) < 1)
    return(NULL)
  if (length(value) < 2)
    return(data)  
  
  # Create list of all dates in dataset as well as list of days missing
  all_days = seq(min(unix_seconds), max(unix_seconds), 86400)
  missing_days = all_days[!all_days %in% unique(unix_seconds)]
  
  # Return result unmodified (except the flag) if no missing days exist
  if (length(missing_days) == 0)
    return(data)
  
  # Create data frame with NA values for missing days, and flag indicating interpolation of value
  missing_data = data.table(
    unix_seconds=missing_days, 
    total_value=rep(NA, length(missing_days)), 
    is_interpolated=rep(T, length(missing_days))
  )
  
  data = rbind(data, missing_data)    
  
  # Interpolate NA values
  data = zoo(data, data$unix_seconds)
  
  # Get the ordered interpolation flag vector back out and
  # do this string comparison because zoo returns TRUE
  # as a string with a leading space (ie as.logical won't parse it correctly)
  is_interpolated = coredata(data)[,'is_interpolated'] != 'FALSE'
  
  data = na.approx(data[,c('unix_seconds', 'total_value')], na.rm=T)
    
  data.table(
    unix_seconds = as.integer(data$unix_seconds),
    total_value = as.numeric(data$total_value),
    is_interpolated = is_interpolated,
    key = 'unix_seconds'
  )
}



METRIC_INFO = NULL
.GetMetricInfo = function(){
  #' Lazy loader for metric metadata
  if (sum(!is.null(METRIC_INFO)) == 0){
    METRIC_INFO <<- meta.getXNetworksMetrics()[,c('id','aggregation_source')]
  }
  METRIC_INFO
}


AGGREGATED_METRICS = NULL
.GetAggregatedMetrics = function(){
  #' Lazy loader for aggregated metric subset
  if (sum(!is.null(AGGREGATED_METRICS)) == 0){    
    AGGREGATED_METRICS <<- subset(.GetMetricInfo(), aggregation_source == 'asset_global')
  }
  AGGREGATED_METRICS
}



