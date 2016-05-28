library(RSQLite)
library(stringr)

source('utils.R')

getWQRawData <- function(start.date, stop.date){
  con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
  d.wq <- dbReadTable(con, 'wmi_water_quality') 
  abbr <- function(x) paste0(str_split(x, ' ')[[1]][1], '...', str_sub(x, -25, -1))
  d.wq <- d.wq %>% 
    mutate(DistributionPointIdentifier = sapply(DistributionPointIdentifier, abbr)) %>%
    mutate(Date=ymd(str_sub(Date, end=10))) %>%
    filter(Date >= start.date & Date <= stop.date)
  d.wq
}

getWQGeoData <- function(d.wq, metric){
  d.geo <- d.wq %>% 
    filter(!is.na(GPSLatitude) & !is.na(GPSLongitude)) %>%
    mutate(GPSLatitude=as.numeric(GPSLatitude)) %>%
    mutate(GPSLongitude=as.numeric(GPSLongitude)) %>%
    group_by(DistributionPointIdentifier, Country, Region, GPSLatitude, GPSLongitude) %>%
    do({
      d <- .
      v <- data.frame(d)[,metric]
      v <- v[is.finite(v)]
      if (length(v) == 0) return(data.frame())
      data.frame(Value=mean(v))
    }) %>% ungroup %>%
    mutate(Size=1 + 10 * (Value - min(Value)) / (max(Value) - min(Value))) %>%
    mutate(Text=sprintf('%s<br>%s/%s', DistributionPointIdentifier, Country, Region))
}

plotAssessmentWQGeoData <- function(d.geo){
  plot_ly(d.geo, lon = GPSLongitude, lat = GPSLatitude, text = Text, color = Country,
          marker = list(size = Size), type = 'scattergeo', autosize=T)
}

getAssessmentIds <- function(d.wq){
  d.wq %>% group_by(AssessmentIdentifier) %>% 
    summarise(Num.DistPoints=length(unique(DistributionPointIdentifier))) %>%
    ungroup %>% arrange(desc(Num.DistPoints)) %>%
    mutate(AssessmentIdLabel=sprintf('%s [%s]', AssessmentIdentifier, Num.DistPoints))
}

getAssessmentWQData <- function(d.wq, assessment.id){
  id.vars <- c('AssessmentIdentifier', 'DistributionPointIdentifier', 'Country')

  wq.vars <- c(
    'WQ_Alkalinity', 'WQ_Conductivity_Log', 'WQ_Chlorine_Free_Log', 
    'WQ_Chlorine_Total_Log', 'WQ_Fecal_Coliforms', 'WQ_Hardness', 
    'WQ_Total_Coliforms', 'WQ_Turbidity_Log'
  )
  
  d.all <- d.wq %>% 
    select(one_of(id.vars), one_of(wq.vars)) %>% 
    melt(id.vars=id.vars, value.name='Value', variable.name='Variable')
  
  d.proj <- d.all %>% filter(AssessmentIdentifier==assessment.id)
  cty <- d.proj$Country[1]
  
  d.cty <- d.all %>% filter(Country == cty)
  
  list(all=d.all, cty=d.cty, proj=d.proj)
}

plotAssessmentWQData <- function(d.proj, type){
  if (!type %in% c('histogram', 'density'))
    stop(sprintf('Plot type "%s" is not valid (must be "histogram" or "density")', type))
  
  if (type == 'histogram') {
    plot_fun <- function(...) geom_histogram(..., bins=10) 
  } else {
    plot_fun <- function(...) geom_density(...)
  }
  
  finite.only <- function(v) v[is.finite(v)]
  
  d.dist <- d.proj$proj %>% 
    group_by(Variable) %>% do({
      d <- .
      
      # Compute range of values for this measurement type globally
      d.all <- d.proj$all %>% filter(Variable == d$Variable[1])
      min.val <- min(finite.only(d.all$Value))
      rng <- max(finite.only(d.all$Value)) - min.val
      
      d %>% na.omit %>%
        mutate(Value=ifelse(Value == -Inf, min.val - rng * .2, Value)) %>%
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
