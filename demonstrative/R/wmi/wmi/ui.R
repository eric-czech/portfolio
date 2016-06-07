#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(plotly)
source('utils.R')

br <- shiny::br

shinyUI(navbarPage("WMI - Project Analysis",
                   
  tabPanel('Water Quality',
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
      sidebarPanel(
        dateInput("wq.start.date", "Start Date:", DEFAULT_START_DATE),
        dateInput("wq.stop.date", "Stop Date:", DEFAULT_STOP_DATE),
        selectizeInput("wq.country", "Country:", choices=NULL),
        selectizeInput("wq.assessment.id", "Project Identifier:", choices=NULL),
        radioButtons("wq.plot.type", "Plot Type", c('density', 'histogram')),
        selectizeInput("wq.metrics", "Water Quality Metrics", 
                       choices=DEFAULT_WQ_METRICS, multiple=T, selected=DEFAULT_WQ_METRICS),
        width=3
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
        tabsetPanel(
          tabPanel("Distribution Points", 
            fluidRow(column(12, selectInput("wq.dist.map.metric", "Metric", c('')))),
            fluidRow(column(12, dataTableOutput("wq.dist.dt"), br(), br())), 
            fluidRow(column(12, plotlyOutput("wq.dist.box.plot", width="1000px"), br(), br())),
            fluidRow(column(12, plotlyOutput("wq.dist.map.plot", width="1000px", height="400px")))
          ),
          tabPanel("Projects Overall", fluidRow(
            column(12, selectInput("wq.proj.map.metric", "Metric", c(''))),
            column(12, dataTableOutput("wq.proj.dt"), br(), br()), 
            column(12, plotlyOutput("wq.proj.box.plot", width="1000px"), br(), br()), 
            column(12, plotlyOutput("wq.proj.map.plot", width="1000px", height="400px"))
          )),
          tabPanel("Single Project", plotOutput("wq.dist.plot", height="400px")),
          tabPanel("Measurement Correlation", plotOutput("wq.corr.plot", height="400px"))
        )
      )
    )
  ), 
  
  tabPanel('Financials',
     # Sidebar with a slider input for number of bins 
     sidebarLayout(
       sidebarPanel(
         dateInput("fi.start.date", "Start Date:", DEFAULT_START_DATE),
         dateInput("fi.stop.date", "Stop Date:", DEFAULT_STOP_DATE),
         selectizeInput("fi.country", "Country:", choices=NULL),
         selectizeInput("fi.assessment.id", "Project Identifier:", choices=NULL),
         radioButtons("fi.plot.type", "Plot Type", c('density', 'histogram')),
         selectizeInput("fi.metrics", "Financial Metrics", 
                        choices=DEFAULT_FI_METRICS, multiple=T, selected=DEFAULT_FI_METRICS),
         width=3
       ),
       
       # Show a plot of the generated distribution
       mainPanel(
         tabsetPanel(
           tabPanel("Projects", fluidRow(
             column(12, selectInput("fi.proj.map.metric", "Metric", c(''))),
             column(12, dataTableOutput("fi.proj.dt")), 
             column(12, plotlyOutput("fi.proj.box.plot", width="1000px")), 
             column(12, plotOutput("fi.proj.cty.ts.plot", width="1000px")), 
             column(12, plotlyOutput("fi.dproj.map.plot", width="1000px", height="400px"))
           )),
           tabPanel("Single Project", plotOutput("fi.proj.plot", height="600px", width="1000px"))
         )
       )
     )
  )
))
