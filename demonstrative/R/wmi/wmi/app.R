library(RSQLite)
library(stringr)
library(dplyr)
library(reshape2)
library(corrplot)
library(DT)
library(scales)
source('utils.R')

#' Prepares measurement values by replacing -Inf measurements 
#' with some sentinel value based on measurement range
prepareMeasurementValues <- function(d){
  wq.vars <- d %>% select(starts_with('WQ_')) %>% names
  
  adjust <- function(measurement){
    # Remove NA values for measurement
    v <- measurement[!is.na(measurement)]
    if (length(v) == 0)
      return(measurement)
    
    if (any(v == Inf)) stop('Found positive infinity measurement (not supported yet)')
    
    # Isolate finite values for measurement
    v.finite <- v[is.finite(v)]
    if (length(v.finite) == 0)
      return(rep(NA, length(measurement)))
    
    # Determine sentinel value to replace -Inf measurements with
    min.val <- min(v.finite) - .1 * (max(v.finite) - min(v.finite))
    v <- ifelse(v == -Inf, min.val, v)
    
    # Return original measurement vector with NA's and finite values unchanged, and 
    # -Inf values replace with sentinel value
    sapply(measurement, function(x) if (is.na(x)) NA else if (x == -Inf) min.val else x)
  }
  for (var in wq.vars){
    d[,var] <- adjust(d[,var])
  }
  d
}

getWQCorPlot <- function(d.wq){
  wq.vars <- d.wq %>% select(starts_with('WQ_')) %>% names
  d.cov <- d.wq %>% select(DistributionPointIdentifier, Country, one_of(wq.vars))
  
  c.var <- wq.vars[!sapply(wq.vars, function(v) all(is.na(d.wq[,v])) || sd(d.wq[,v], na.rm=T) == 0)]
  impute <- function(x) ifelse(is.na(x), mean(x, na.rm=T), x)
  d.wq[,c.var] %>% mutate_each(funs(impute)) %>% cor %>% corrplot
}

getWQRawData <- function(start.date, stop.date){
  con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
  d.wq <- dbReadTable(con, 'wmi_water_quality') 
  abbr <- function(x) paste0(str_split(x, ' ')[[1]][1], '...', str_sub(x, -25, -1))
  d.wq <- d.wq %>% 
    mutate(DistributionPointIdentifier = sapply(DistributionPointIdentifier, abbr)) %>%
    mutate(Date=ymd(str_sub(Date, end=10))) %>%
    filter(Date >= start.date & Date <= stop.date)
  d.wq <- prepareMeasurementValues(d.wq)
  d.wq
}

getWQGeoData <- function(d.wq, metrics, group.cols, text.gen){
  c.meta <- unique(c(group.cols, 'Lat', 'Lon'))
  d.geo <- d.wq %>% 
    
    # Convert lat/lon values to float
    rename(Lat=GPSLatitude, Lon=GPSLongitude) %>%
    mutate(Lat=as.numeric(Lat), Lon=as.numeric(Lon)) %>% 
    
    # Pivot water quality measurements into rows
    select(one_of(c.meta), one_of(metrics)) %>%
    melt(id.vars=c.meta, variable.name='Variable', value.name='Value') %>% 
    
    # Group by selected columns and determine aggregate measurement value for each group
    group_by_(.dots=c(group.cols, 'Variable')) %>% do({
      d <- .
          
      # Determine location centroid
      get.geo.center <- function(x) median(x[!is.na(x)])
      lat <- get.geo.center(d$Lat)
      lon <- get.geo.center(d$Lon)
      data.frame(Lat=lat, Lon=lon, Value=mean(d$Value, na.rm=T))
    }) %>% ungroup %>% 
    
    # Remove records for locations with no associated value
    filter(!is.na(Value)) %>%
    
    # Determine size for plotting as value on 2-15 scale
    group_by(Variable) %>%
    mutate(Size=2 + 13 * (Value - min(Value)) / (max(Value) - min(Value))) %>% 
    ungroup %>%
    
    # Add text description of each record
    ungroup %>% text.gen %>%
    
    # Add variable id, which is useful for map generation
    mutate(VariableId=as.integer(Variable), Variable=as.character(Variable))
}

plotWQGeoData <- function(d.geo, title=''){
  d.geo %>%
    filter(!is.na(Lat) & !is.na(Lon)) %>%
    plot_ly(
      lon = Lon, lat = Lat, text = Text, color = Country,
      shape=Variable, marker = list(size = Size), type = 'scattergeo'
    ) %>% 
    layout(title=title)
}

getWQAssessmentIds <- function(d.wq){
  d.wq %>% group_by(AssessmentIdentifier) %>% 
    summarise(Num.DistPoints=length(unique(DistributionPointIdentifier))) %>%
    ungroup %>% arrange(desc(Num.DistPoints)) %>%
    mutate(AssessmentIdLabel=sprintf('%s [%s]', AssessmentIdentifier, Num.DistPoints))
}

getFIAssessmentIds <- function(d.fi){
  d.fi %>% group_by(AssessmentIdentifier) %>% 
    summarise(Num.Months=length(unique(Date))) %>%
    ungroup %>% arrange(desc(Num.Months)) %>%
    mutate(AssessmentIdLabel=sprintf('%s [%s]', AssessmentIdentifier, Num.Months))
}

getAssessmentWQData <- function(d.wq, assessment.id, metrics){
  id.vars <- c('AssessmentIdentifier', 'DistributionPointIdentifier', 'Country')
  
  d.all <- d.wq %>% 
    select(one_of(id.vars), one_of(metrics)) %>% 
    melt(id.vars=id.vars, value.name='Value', variable.name='Variable') %>%
    filter(!is.na(Value))
  
  d.proj <- d.all %>% filter(AssessmentIdentifier==assessment.id)
  cty <- d.proj$Country[1]
  
  d.cty <- d.all %>% filter(Country == cty)
  
  list(all=d.all, cty=d.cty, proj=d.proj)
}

# plotWQAlerts() <- function(d.proj)

plotWQProjectDistribution <- function(d.proj, type){
  if (!type %in% c('histogram', 'density'))
    stop(sprintf('Plot type "%s" is not valid (must be "histogram" or "density")', type))
  
  if (type == 'histogram') {
    plot_fun <- function(...) geom_histogram(..., bins=10) 
  } else {
    plot_fun <- function(...) geom_density(...)
  }
  
  d.dist <- d.proj$proj %>% 
    group_by(Variable) %>% do({
      d <- .
      
      # Compute range of values for this measurement type globally
      d.all <- d.proj$all %>% filter(Variable == d$Variable[1])
      rng <- max(d.all$Value) - min(d.all$Value)
      
      d %>% filter(!is.na(Value)) %>%
        mutate(Value=Value + rnorm(length(Value), 0, rng * .01)) # Add jitter for vertical lines
    }) %>% ungroup
  
  print(d.dist %>% arrange(Variable) %>% data.frame)
  ggplot(NULL) +
    plot_fun(data=d.proj$all %>% filter(is.finite(Value)), aes(x=Value, fill='Global'), alpha=.3) +
    plot_fun(data=d.proj$cty %>% filter(is.finite(Value)), aes(x=Value, fill='Country'), alpha=.3) +
    geom_vline(data=d.dist, aes(xintercept=Value, color=DistributionPointIdentifier)) +
    facet_wrap(~Variable, scales='free', ncol=2) + 
    scale_fill_brewer(palette = "Set2") +
    theme_bw()
}


##### Financial Data #####

getFIRawData <- function(start.date, stop.date){
  con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
  d.fi <- dbReadTable(con, 'wmi_project_timelines') 
  abbr <- function(x) paste0(str_split(x, ' ')[[1]][1], '...', str_sub(x, -25, -1))
  d.fi <- d.fi %>% 
    mutate(Date=ymd(str_sub(Date, end=10))) %>%
    filter(Date >= start.date & Date <= stop.date)
  d.fi
}

getAssessmentFIData <- function(d.wq, assessment.id, metrics){
  id.vars <- c('AssessmentIdentifier', 'Date', 'Interpolated', 'Country')
  
  d.all <- d.wq %>% 
    select(one_of(id.vars), one_of(metrics)) %>% 
    melt(id.vars=id.vars, value.name='Value', variable.name='Variable') %>%
    filter(!is.na(Value))
  
  d.proj <- d.all %>% filter(AssessmentIdentifier==assessment.id)
  cty <- d.proj$Country[1]
  
  d.cty <- d.all %>% filter(Country == cty)
  
  list(all=d.all, cty=d.cty, proj=d.proj)
}

plotFIProjectDistribution <- function(d.proj, type){
  # if (!type %in% c('histogram', 'density'))
  #   stop(sprintf('Plot type "%s" is not valid (must be "histogram" or "density")', type))
  # 
  # if (type == 'histogram') {
  #   plot_fun <- function(...) geom_histogram(..., bins=10) 
  # } else {
  #   plot_fun <- function(...) geom_density(...)
  # }
  
  # d.proj$cty %>% group_by(Date) %>% summarise()
  # d.proj$cty %>% ggplot(aes(x=Date, y=Value)) + geom_smooth()
  # browser()
  # save(d.proj, file='/tmp/dproj.Rdata')
  # e <- new.env()
  # load(file='/tmp/dproj.Rdata', envir = e)
  # d.proj <- e$d.proj
  # print(d.proj$proj %>% arrange(Variable) %>% data.frame)
  
  d.proj$proj <- d.proj$proj %>% 
    mutate(IsEstimate=factor(Interpolated, levels=c(0, 1), labels=c('No', 'Yes')))
  
  ggplot(NULL) +
    #plot_fun(data=d.proj$all %>% filter(is.finite(Value)), aes(x=Value, fill='Global'), alpha=.3) +
    geom_smooth(data=d.proj$cty, aes(x=Date, y=Value, color='Country Benchmark')) +
    geom_line(data=d.proj$proj, aes(x=Date, y=Value, color=AssessmentIdentifier)) +
    geom_point(data=d.proj$proj, aes(x=Date, y=Value, shape=IsEstimate)) +
    facet_wrap(~Variable, ncol=1, scales='free_y') + 
    scale_color_discrete(guide=guide_legend(title='Timeseries')) +
    scale_fill_brewer(palette = "Set2") +
    theme_bw() + 
    ggtitle('Project Performance (vs Country Benchmark)')
}
