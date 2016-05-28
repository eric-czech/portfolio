#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
source('app.R')

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
   
  dateRangeData <- reactive({
    start.date <- input$start.date
    stop.date <- input$stop.date
    if (is.null(start.date))
      start.date <- DEFAULT_START_DATE
    if (is.null(stop.date))
      stop.date <- DEFAULT_STOP_DATE
    c(start.date, stop.date)
  })
  
  wqData <- reactive({
    dates <- dateRangeData()
    d.wq <- getWQRawData(dates[1], dates[2])
    
    if (str_length(input$country) > 0){
      print(paste('filtering to ', input$country))
      d.id <- getAssessmentIds(d.wq %>% filter(Country == input$country))
    } else {
      d.id <- getAssessmentIds(d.wq)
    }
    
    list(d.wq=d.wq, d.id=d.id)
  })
  
  observe({
    d <- wqData()
    ctys <- d$d.wq$Country %>% unique
    ctys <- as.list(ctys) %>% setNames(ctys)
    ids <- as.list(d$d.id$AssessmentIdentifier) %>% setNames(d$d.id$AssessmentIdLabel)
    
    updateSelectizeInput(session, 'country', choices=ctys, selected=input$country)
    updateSelectizeInput(session, 'assessment.id', choices=ids, selected=input$assessment.id)
  })

  
  output$wq.dist.plot <- renderPlot({
    d <- wqData()
    d.proj <- getAssessmentWQData(d$d.wq, input$assessment.id)
    if (nrow(d.proj$proj) == 0)
      return(NULL)
    plotAssessmentWQData(d.proj, input$wq.plot.type)
  })
  
  output$wq.map.plot <- renderPlotly({
    metric <- 'WQ_Alkalinity'
    d <- wqData()
    d.geo <- getWQGeoData(d$d.wq, metric)
    plotAssessmentWQGeoData(d.geo)
  })
  
})
