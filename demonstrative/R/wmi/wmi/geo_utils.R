
correctMistakenCoordinates <- function(d){
  clean <- function(id, cty, lat, lon){
    def <- c(NA, NA)
    if (any(is.null(c(lat, lon)))) return(def)
    
    res <- c(lat, lon)
    if (cty == 'Peru'){
      res <- c(-abs(lat), -abs(lon))
    } else if (cty == 'Haiti'){
      res <- c(abs(lat), -abs(lon))
    } else if (cty == 'Indonesia'){
      res <- c(lat, abs(lon))
    } else if (cty %in% c('Mexico', 'Honduras')){
      res <- c(lat, -abs(lon))
    } else if (cty == 'Liberia'){
      res <- c(abs(lat), lon)
    } else if (cty == 'Malawi'){
      res <- c(-abs(lat), abs(lon))
    } else if (cty %in% c('Tanzania', 'Uganda', 'Kenya')){
      res <- c(lat, abs(lon))
    }
    res
  }
  res <- d %>% 
    rowwise() %>% 
    mutate(Lat=clean(AssessmentID, Country, Lat, Lon)[1]) %>% 
    mutate(Lon=clean(AssessmentID, Country, Lat, Lon)[2]) 
  res
}