#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

source('utils.R')

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Prioritizing Cars to Buy (or race ... or something)"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      dateInput("start.date", "Start Date:", value=DEFAULT_START_DATE),
      dateInput("stop.date", "Stop Date:"),
      sliderInput("wt.mpg", "MPG Weight:", min = 0, max = 100, value = 50),
      sliderInput("wt.hp", "Horsepower Weight:", min = 0, max = 100, value = 50),
      sliderInput("wt.gear", "Num Gears Weight:", min = 0, max = 100, value = 50),
      actionButton("calcButton", "Calculate")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotlyOutput("rankingPlot", height='400px'),
      plotlyOutput("diagnosticsPlot", height='400px')
    )
  )
))
