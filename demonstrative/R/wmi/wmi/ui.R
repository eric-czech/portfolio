#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("WMI - Project Analysis"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      selectizeInput("country", "Country:", choices=NULL),
      selectizeInput("assessment.id", "Project Identifier:", choices=NULL),
      radioButtons("wq.plot.type", "Plot Type", c('density', 'histogram'))
    ),
  
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(
        tabPanel("WQ - Distributions", plotOutput("wq.dist.plot", height="600px")),
        tabPanel("WQ - Maps", plotlyOutput("wq.map.plot", height="600px"))
      )
    )
  )
))
