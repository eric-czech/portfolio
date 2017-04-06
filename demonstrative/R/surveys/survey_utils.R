#' @title Histogram generator for survey response data compatible with "survey" package
svy.hist <- function(formula, design, breaks='Sturges', right=T, include.lowest = T, integrate=T){
  mf <- model.frame(formula, model.frame(design), na.action=na.pass)
  if (ncol(mf) > 1) stop("Only one variable allowed.")
  variable<-mf[,1]
  varname<-names(mf)
  h <- hist(variable, plot=FALSE, breaks=breaks, right=right)
  props <- coef(svymean(~cut(variable, h$breaks, right=right, include.lowest=include.lowest),
                        design, na.rm=TRUE))
  if (integrate)
    h$density<-props
  else
    h$density<-props/diff(h$breaks)
  h$counts <- props*sum(weights(design,"sampling"))
  h
}

# weighted.hist <- function(d, wt, breaks='Sturges', right=T, include.lowest = T, integrate=T){
#   
#   variable <- mf[,1]
#   h <- hist(variable, plot=FALSE, breaks=breaks,right=right)
#   props <- coef(svymean(~cut(variable, h$breaks, right=right, include.lowest=include.lowest),
#                         design, na.rm=TRUE))
#   if (integrate)
#     h$density<-props
#   else
#     h$density<-props/diff(h$breaks)
#   h$counts <- props*sum(weights(design,"sampling"))
#   h
# }