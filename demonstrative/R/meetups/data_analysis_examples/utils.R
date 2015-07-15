

get.raw.data <- function() {
  csv <- '/Users/eczech/repos/portfolio/demonstrative/R/meetups/data_analysis_examples/data/crime_data.csv'
  data <- read.csv(csv, stringsAsFactors=F)
  
  
  data <- data %>% 
    melt(id.vars='Country', variable.name = 'Year', value.name = 'Homicide.Rate') %>% 
    mutate(Year = as.numeric(str_replace(Year, 'X', '')) - 2000) %>%
    filter(!is.na(Homicide.Rate))
}