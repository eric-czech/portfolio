source('~/NBS-DS/R/impala/transforms.R')

# Unit tests for impala/transforms.R functions

#' Tests the validity of timeseries unstacking for metrics
#' with only total values (e.g. Facebook Page Likes)
test.UnstackTimeseries.TotalsOnly = function() {
  InterpolationValidation(11)
  InterpolationValidation(28)
}

#' Tests the validity of timeseries unstacking for metrics
#' with only delta values (e.g. iTunes Track Units)
test.UnstackTimeseries.DeltasOnly = function() {
  DeltaValidation(133)
  DeltaValidation(100)
}

#' Tests the validity of timeseries unstacking for metrics
#' with pre-aggregated delta and total values (e.g. YouTube Video Views)
test.UnstackTimeseries.PreAggregated = function() {
  AggregateValidation(44)
  AggregateValidation(42)
}

InterpolationValidation = function(.metric_id) {
  # Create a data frame with data present for 3 days, then missing for 2, then present for 3 more
  data = GetBaseDf(.metric_id, 't', 1:3, '20140501', '20140503')  
  data = rbind(data, GetBaseDf(.metric_id, 't', 6:8, '20140506', '20140508'))

  # Unstack the timeseries
  data = with(data, .UnstackTimeseries(metric_id, count_type, unix_seconds, value))
  
  # Validate 8 records, with the last delta being NA, all other deltas being 1,
  # and all totals being an in order sequence from 1 to 8
  checkEquals(nrow(data), 8)
  checkEquals(sum(data$delta_value == 1, na.rm=T), 7)
  checkEquals(data$total_value, seq(1:8))
  checkTrue(is.na(data$delta_value[nrow(data)]))
}

DeltaValidation = function(.metric_id) {
  # Create a data frame with data present for 3 days, then missing for 2, then present for 3 more
  data = GetBaseDf(.metric_id, 'd', 1:3, '20140501', '20140503')  
  data = rbind(data, GetBaseDf(.metric_id, 'd', 6:8, '20140506', '20140508'))

  # Unstack the timeseries
  data = with(data, .UnstackTimeseries(metric_id, count_type, unix_seconds, value))
  
  # Validate 6 records, with the last delta being NA, other deltas
  # being unmodified and totals all being NA
  checkEquals(nrow(data), 6)
  checkEquals(sum(data$delta_value, na.rm=T), 19)
  checkEquals(sum(is.na(data$total_value)), 6)
  checkTrue(is.na(data$delta_value[nrow(data)]))
}



AggregateValidation = function(.metric_id) {
  # Create a data frame with data present for 3 days, then missing for 2, then present for 3 more
  # for both delta and total values
  data = GetBaseDf(.metric_id, 'd', 1:3, '20140501', '20140503')  
  data = rbind(data, GetBaseDf(.metric_id, 'd', 6:8, '20140506', '20140508'))
  data = rbind(data, GetBaseDf(.metric_id, 't', 1:3, '20140501', '20140503'))
  data = rbind(data, GetBaseDf(.metric_id, 't', 6:8, '20140506', '20140508'))
  
  # Unstack the timeseries
  data = with(data, .UnstackTimeseries(metric_id, count_type, unix_seconds, value))
  
  # Validate 6 records, with the last delta being NA, other deltas
  # being unmodified and totals all being NA
  checkEquals(nrow(data), 6)
  checkEquals(data$delta_value, c(1:3, 6:7, NA))
  checkEquals(data$total_value, c(1:3, 6:8))
  checkTrue(is.na(data$delta_value[nrow(data)]))
}

# Returns frame with data on given time range
GetBaseDf = function(.metric_id, .count_type, values, start_date, end_date){
  data.frame(
    metric_id = rep(.metric_id, 3),
    count_type = rep(.count_type, 3),
    unix_seconds = seq(as.integer(ymd(start_date)), as.integer(ymd(end_date)), 86400),
    value = values
  )
}


