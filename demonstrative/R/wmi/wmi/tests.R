library(RSQLite)
library(dplyr)
library(plotly)
library(reshape2)

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

metric <- 'WQ_Alkalinity'
d.geo <- d %>% 
  filter(!is.na(GPSLatitude) & !is.na(GPSLongitude)) %>%
  mutate(GPSLatitude=as.numeric(GPSLatitude)) %>%
  mutate(GPSLongitude=as.numeric(GPSLongitude)) %>%
  group_by(DistributionPointIdentifier, AssessmentID, Country, Region, GPSLatitude, GPSLongitude) %>%
  do({
    d <- .
    v <- data.frame(d)[,metric]
    v <- v[is.finite(v)]
    if (length(v) == 0) return(data.frame())
    data.frame(Value=mean(v))
  }) %>% ungroup %>%
  mutate(Size=1 + 10 * (Value - min(Value)) / (max(Value) - min(Value))) %>%
  mutate(Text=sprintf('%s<br>%s/%s', DistributionPointIdentifier, Country, Region))

plot_ly(d.geo, lon = GPSLongitude, lat = GPSLatitude, text = Text, color=Country,
        marker = list(size = Size), type = 'scattergeo', autosize=T)

##### Financials #####
con = dbConnect(drv=SQLite(), dbname='/Users/eczech/data/research/wmi/data/wmi.db')
d <- dbReadTable(con, 'wmi_project_timelines')


metrics <- c('IncomeNet', 'ExpensesAll', 'IncomeAll', 'QuantityWaterAll')
meta <- c('AssessmentIdentifier', 'Continent', 'Country', 'Date')
