source('~/NBS-DS/R/impala/package.R')

options(error=traceback)
options(warn=1)
# Performance tests for R+impala query transformation library

# Select statement for all data for artists below some id on the time range 2014-01-01 to 2014-05-26
query_format = 'SELECT * FROM idx_entity WHERE entity_id < %s and unix_seconds between 1388534400 and 1401062400'

# Takes ~20 seconds, returns ~400k rows, takes 24 seconds to unstack
system.time(metric.data <- impalaQuery(sprintf(query_format, 100), col.names=COLS_IDX_ENTITY))
system.time(unstacked <- UnstackTimeseries(metric.data))

# Takes ~140 seconds, returns ~3.3M rows, takes 197 seconds to unstack
system.time(metric.data <- impalaQuery(sprintf(query_format, 1000), col.names=COLS_IDX_ENTITY))
system.time(unstacked <- UnstackTimeseries(metric.data))

# Takes ~993 seconds (only 400s user), returns ~22.2M rows, takes 1288 seconds to unstack
system.time(metric.data <- impalaQuery(sprintf(query_format, 10000), col.names=COLS_IDX_ENTITY))
system.time(unstacked <- UnstackTimeseries(metric.data))



