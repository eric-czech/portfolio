library(RSQLite)
library(dplyr)
library(plotly)
library(reshape2)

source('app.R')

##### Water Quality #####

con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
d <- dbReadTable(con, 'wmi_water_quality')

d %>% group_by(AssessmentIdentifier) %>% 
  summarise(n=length(unique(DistributionPointIdentifier))) %>%
  arrange(desc(n)) %>% head(25)

#assessment.id <- 'San Quintin (Ocosingo, Chiapas - MX) (1086)'
assessment.id <- 'Kaporo Extension (2918)'

id.vars <- c('AssessmentIdentifier', 'DistributionPointIdentifier', 'Country')
# wq.vars <- d %>% select(starts_with('WQ_')) %>% names
wq.vars <- c(
  'WQ_Alkalinity', 'WQ_Conductivity_Log', 'WQ_Chlorine_Free_Log', 
  'WQ_Chlorine_Total_Log', 'WQ_Fecal_Coliforms', 'WQ_Hardness', 
  'WQ_Total_Coliforms', 'WQ_Turbidity_Log'
)

d.all <- d %>% 
  select(one_of(id.vars), one_of(wq.vars)) %>% 
  melt(id.vars=id.vars, value.name='Value', variable.name='Variable') %>% 
  na.omit
  
d.proj <- d.all %>% filter(AssessmentIdentifier==assessment.id)
cty <- d.proj$Country[1]

d.cty <- d.all %>% filter(Country == cty)

ggplot(NULL) +
  geom_density(data=d.all, aes(x=Value, fill='Global'), alpha=.3) +
  geom_density(data=d.cty, aes(x=Value, fill='Country'), alpha=.3) +
  geom_vline(data=d.proj, aes(xintercept=Value, color=DistributionPointIdentifier)) +
  facet_wrap(~Variable, scales='free', ncol=1) + 
  scale_fill_brewer(palette = "Set2") +
  theme_bw()


d.metric %>% group_by(Country, Variable) %>% summarise(Country.Value=mean(Value))
d.metric %>% na.omit %>% 
  ggplot(aes(x=value)) + geom_density() + 
  facet_wrap(~Country, scales='free_y')

# Maps

n.lat.lon <- d %>% group_by(DistributionPointIdentifier) %>% 
  summarise(n=length(unique(paste(GPSLatitude, GPSLongitude)))) %>%
  .$n %>% unique
if (n.lat.lon != 1) stop('Found conflicting lat/lon pairs for same distribution point')

#metrics <- c('WQ_Alkalinity', 'WQ_Turbidity_Log')
metrics <- d %>% select(starts_with('WQ_')) %>% names

# By project
group.cols <- c('AssessmentIdentifier', 'Country', 'Region')
text.gen <- function(d) d %>% 
  mutate(Text=sprintf('%s<br>%s/%s', AssessmentIdentifier, Country, Region))

# By distribution point
group.cols <- c('DistributionPointIdentifier', 'Country', 'Region')
text.gen <- function(d) d %>% 
  mutate(Text=sprintf('%s<br>%s/%s', DistributionPointIdentifier, Country, Region))

c.meta <- unique(c(group.cols, 'Lat', 'Lon'))
d.geo <- d %>% 
  
  # Convert lat/lon values to float
  rename(Lat=GPSLatitude, Lon=GPSLongitude) %>%
  mutate(Lat=as.numeric(Lat), Lon=as.numeric(Lon)) %>% 
  
  # Pivot water quality measurements into rows
  select(one_of(c.meta), one_of(metrics)) %>%
  melt(id.vars=c.meta, variable.name='Variable', value.name='Value') %>% 
  
  # Determine min and max finite values for each measurement type
  group_by(Variable) %>% 
  mutate(MaxValue=max(Value[is.finite(Value)]), MinValue=min(Value[is.finite(Value)])) %>%
  ungroup %>% filter(is.finite(MaxValue) & is.finite(MinValue)) %>% 
  
  # Group by selected columns and determine aggregate measurement value for each group
  group_by_(.dots=c(group.cols, 'Variable')) %>% do({
    d <- .
    
    # Get non-na values for measurements
    v <- d$Value[!is.na(d$Value)]
    if (length(v) == 0)
      return(data.frame())
    
    # Replace -Inf measurements with sentinel value
    min.val <- d$MinValue[1] - .1 * (d$MaxValue[1] - d$MinValue[1])
    v <- ifelse(v == -Inf, min.val, v)
    if (!all(is.finite(v))) stop('Found non-finite measurements in group after dealing with -Inf values')
    
    # Determine location centroid
    get.geo.center <- function(x) median(x[!is.na(x)])
    lat <- get.geo.center(d$Lat)
    lon <- get.geo.center(d$Lon)
    
    # Return mean of measurement values for group
    data.frame(Value=mean(v), Lat=lat, Lon=lon)
  }) %>% ungroup %>%
  
  # Rescale measurement values to 1-15 scale
  group_by(Variable) %>%
  mutate(Size=1 + 14 * (Value - min(Value)) / (max(Value) - min(Value))) %>% 
  ungroup %>%
  
  # Add text description of each record
  ungroup %>% text.gen %>%
  
  # Add variable id, which is useful for map generation
  mutate(VariableId=as.integer(Variable), Variable=as.character(Variable))
  

var <- metrics[2]
d.geo %>%
  filter(Variable==var) %>%
  filter(!is.na(Lat) & !is.na(Lon)) %>%
  plot_ly(
    lon = Lon, lat = Lat, text = Text, 
    marker = list(size = Size), 
    color=Country,
    type = 'scattergeo',
    showlegend=T
  ) %>% layout(title=var)


# WQ Covariance Analysis

d.all <- prepareMeasurementValues(d)
wq.vars <- d.all %>% select(starts_with('WQ_')) %>% names
d.cov <- d.all %>% select(DistributionPointIdentifier, Country, one_of(wq.vars))

c.var <- wq.vars[!sapply(wq.vars, function(v) all(is.na(d.cov[,v])) || sd(d.cov[,v], na.rm=T) == 0)]
impute <- function(x) ifelse(is.na(x), mean(x, na.rm=T), x)
d.cov[,c.var] %>% mutate_each(funs(impute)) %>% cor %>% corrplot
pairs(d.cov[,c.var])
cor(d.cov[,wq.vars], use='pairwise.complete.obs')



##### Financials #####

con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
d <- dbReadTable(con, 'wmi_project_timelines')


metrics <- c('IncomeNet', 'ExpensesAll', 'IncomeAll', 'QuantityWaterAll')
meta <- c('AssessmentIdentifier', 'Continent', 'Country', 'Date')
