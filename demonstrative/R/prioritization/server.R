#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(plotly)
library(ggplot2)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  getData <- eventReactive(input$calcButton, {
    d <- mtcars
    names(d) <- sapply(names(d), toupper)
    
    wts <- c(
      mpg = input$wt.mpg,
      hp = input$wt.hp,
      gear = input$wt.gear
    )
    wts <- wts / sum(wts)
    
    
    scale <- function(x) (x - min(x)) / (max(x) - min(x))
    d.score <- d %>% 
      add_rownames(var='Vehicle.Make') %>%
      select(Vehicle.Make, MPG, HP, GEAR) %>%
      mutate_each(funs(scale), -Vehicle.Make) %>%
      mutate(MPG = MPG * wts['mpg'], HP = HP * wts['hp'], GEAR = GEAR * wts['gear']) %>%
      mutate(Priority.Score=MPG + HP + GEAR)
    list(d.score, wts)
  })
  
  output$diagnosticsPlot <- renderPlotly({
    d <- getData()
    d.score <- d[[1]]
    wts <- d[[2]]
    d.score %>% 
      select(-Priority.Score) %>%
      melt(id.vars=c('Vehicle.Make'), value.name='Value', variable.name='Variable') %>%
      plot_ly(x=Value, color=Variable, type='histogram', opacity=.8) %>%
      layout(
        title='Variable Contribution Distribution',
        margin=list(b=150)
      )
  })
  
  output$rankingPlot <- renderPlotly({
    d <- getData()
    d.score <- d[[1]]
    wts <- d[[2]]
    
    d.score %>% 
      arrange(desc(Priority.Score)) %>% head(10) %>%
      melt(id.vars=c('Vehicle.Make', 'Priority.Score'), value.name='Value', variable.name='Variable') %>%
      arrange(Priority.Score) %>%
      plot_ly(
        y=Vehicle.Make, x=Value, color=Variable, 
        type='bar', orientation='h', 
        text = paste("Score =", round(Priority.Score, 2))
      ) %>%
      layout(
        title='Best Cars',
        barmode='stack', margin=list(l=150, b=120)
      )
  })
  
})
