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
   
  wqDateRangeData <- reactive({
    start.date <- input$wq.start.date
    stop.date <- input$wq.stop.date
    if (is.null(start.date))
      start.date <- DEFAULT_START_DATE
    if (is.null(stop.date))
      stop.date <- DEFAULT_STOP_DATE
    c(start.date, stop.date)
  })
  
  wqData <- reactive({
    dates <- wqDateRangeData()
    d.wq <- getWQRawData(dates[1], dates[2]) 
    d.wq <- prepareMeasurementValues(d.wq)
    
    if (str_length(input$wq.country) > 0){
      print(paste('filtering to ', input$wq.country))
      d.id <- getWQAssessmentIds(d.wq %>% filter(Country == input$wq.country))
    } else {
      d.id <- getWQAssessmentIds(d.wq)
    }
    
    list(d.wq=d.wq, d.id=d.id)
  })
  
  observe({
    d <- wqData()
    ctys <- d$d.wq$Country %>% unique
    ctys <- as.list(ctys) %>% setNames(ctys)
    ids <- as.list(d$d.id$AssessmentIdentifier) %>% setNames(d$d.id$AssessmentIdLabel)
    metrics <- d$d.wq %>% select(starts_with('WQ_')) %>% names
    metrics <- as.list(metrics) %>% setNames(metrics)
    
    updateSelectizeInput(session, 'wq.country', choices=ctys, selected=input$wq.country)
    updateSelectizeInput(session, 'wq.assessment.id', choices=ids, selected=input$wq.assessment.id)
    updateSelectizeInput(session, 'wq.metrics', choices=metrics, selected=input$wq.metrics)
    updateSelectInput(session, 'wq.dist.map.metric', choices=input$wq.metrics, selected=input$wq.map.metric)
    updateSelectInput(session, 'wq.proj.map.metric', choices=input$wq.metrics, selected=input$wq.map.metric)
    
    d <- fiData()
    ctys <- d$d.fi$Country %>% unique
    ctys <- as.list(ctys) %>% setNames(ctys)
    ids <- as.list(d$d.id$AssessmentIdentifier) %>% setNames(d$d.id$AssessmentIdLabel)
    metrics <- d$d.fi %>% 
      select(starts_with('Expenses'), starts_with('Income'), starts_with('Cash')) %>% names
    metrics <- as.list(metrics) %>% setNames(metrics)
    
    updateSelectizeInput(session, 'fi.country', choices=ctys, selected=input$fi.country)
    updateSelectizeInput(session, 'fi.assessment.id', choices=ids, selected=input$fi.assessment.id)
    updateSelectizeInput(session, 'fi.metrics', choices=metrics, selected=input$fi.metrics)
    updateSelectInput(session, 'fi.proj.map.metric', choices=input$fi.metrics, selected=input$fi.map.metric)
  })

  
  output$wq.dist.plot <- renderPlot({
    d <- wqData()
    metrics <- input$wq.metrics
    if (is.null(metrics) || length(metrics) == 0)
      return(NULL)
    d <- getAssessmentWQData(d$d.wq, input$wq.assessment.id, metrics)
    if (nrow(d$proj) == 0)
      return(NULL)
    plotWQProjectDistribution(d, input$wq.plot.type)
  })
  
  output$wq.dist.dt <- renderDataTable({
    metric <- input$wq.dist.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()
    group.cols <- c(
      'DistributionPointID', 'DistributionPointName', 
      'AssessmentID', 'AssessmentName', 
      'Country', 'Region'
    )
    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, function(d) d %>% mutate(Text=''))
    datatable(d.geo %>% arrange(desc(Value)) %>% select(one_of(group.cols), Value))
  })
  
  output$wq.dist.box.plot <- renderPlotly({
    metric <- input$wq.dist.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()
    group.cols <- c('DistributionPointIdentifier', 'Country', 'Region')
    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, function(d) d %>% mutate(Text=''))
    d.geo %>% 
      plot_ly(type='box', x=Country, y=Value, boxmean=T) %>%
      layout(xaxis=list(title=''), title='Value By Country')
  })
  
  output$wq.dist.map.plot <- renderPlotly({
    metric <- input$wq.dist.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()
    
    group.cols <- c('DistributionPointIdentifier', 'Country', 'Region')
    text.gen <- function(d) d %>% 
      mutate(Text=sprintf('Dist Point: %s<br>Location: %s / %s<br>Value: %.3f', 
                          DistributionPointIdentifier, Country, Region, Value))
    
    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, text.gen) %>% filter(!is.na(Value))
    plotWQGeoData(d.geo, title='Value by Distribution Point')
  })
  
  output$wq.proj.dt <- renderDataTable({
    metric <- input$wq.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()
    group.cols <- c(
      'AssessmentID', 'AssessmentName', 
      'Country', 'Region'
    )
    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, function(d) d %>% mutate(Text=''))
    datatable(d.geo %>% arrange(desc(Value)) %>% select(one_of(group.cols), Value))
  })
  
  output$wq.proj.box.plot <- renderPlotly({
    metric <- input$wq.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()
    group.cols <- c('AssessmentIdentifier', 'Country', 'Region')
    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, function(d) d %>% mutate(Text=''))
    d.geo %>% 
      plot_ly(type='box', x=Country, y=Value, boxmean=T) %>%
      layout(xaxis=list(title=''), title='Value By Country')
  })
  
  output$wq.proj.map.plot <- renderPlotly({
    metric <- input$wq.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- wqData()

    group.cols <- c('AssessmentIdentifier', 'Country', 'Region')
    text.gen <- function(d) d %>%
      mutate(Text=sprintf('Assessment: %s<br>Location: %s / %s<br>Value: %.3f', 
                          AssessmentIdentifier, Country, Region, Value))

    d.geo <- getWQGeoData(d$d.wq, metric, group.cols, text.gen) %>% filter(!is.na(Value))
    plotWQGeoData(d.geo, title='Value by Project')
  })
  
  output$wq.corr.plot <- renderPlot({
    d <- wqData()
    getWQCorPlot(d$d.wq)
  })
  
  ##### Financials #####
  
  fiDateRangeData <- reactive({
    start.date <- input$fi.start.date
    stop.date <- input$fi.stop.date
    if (is.null(start.date))
      start.date <- DEFAULT_START_DATE
    if (is.null(stop.date))
      stop.date <- DEFAULT_STOP_DATE
    c(start.date, stop.date)
  })
  
  fiData <- reactive({
    dates <- fiDateRangeData()
    d.fi <- getFIRawData(dates[1], dates[2]) 
    
    if (str_length(input$fi.country) > 0){
      print(paste('filtering to ', input$fi.country))
      d.id <- getFIAssessmentIds(d.fi %>% filter(Country == input$fi.country))
    } else {
      d.id <- getFIAssessmentIds(d.fi)
    }
    
    list(d.fi=d.fi, d.id=d.id)
  })
  
  output$fi.proj.dt <- renderDataTable({
    metric <- input$fi.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- fiData()
    group.cols <- c('AssessmentID', 'AssessmentName', 'Country', 'Region')
    d$d.fi %>% select(one_of(c(group.cols, metric))) %>%
      rename_(Value=metric) %>%
      filter(!is.na(Value)) %>%
      arrange(desc(Value)) %>%
      datatable
  })
  
  output$fi.proj.box.plot <- renderPlotly({
    metric <- input$fi.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- fiData()
    group.cols <- c('AssessmentID', 'AssessmentName', 'Country', 'Region')
    d$d.fi %>% select(one_of(c(group.cols, metric))) %>%
      rename_(Value=metric) %>%
      filter(!is.na(Value)) %>%
      plot_ly(type='box', x=Country, y=Value, boxmean=T) %>%
      layout(xaxis=list(title=''), title='Distribution By Country')
  })
  
  output$fi.proj.cty.ts.plot <- renderPlot({
    metric <- input$fi.proj.map.metric
    if (str_length(metric) == 0)
      return(NULL)
    d <- fiData()
    group.cols <- c('AssessmentID', 'AssessmentName', 'Country', 'Date')

    # d$d.fi %>% select(one_of(c(group.cols, metric))) %>%
    #   rename_(Value=metric) %>%
    #   group_by(Country, Date) %>% summarise(Value=mean(Value, na.rm=T)) %>%
    #   plot_ly(x=Date, y=Value, color=Country, pallette='Set1') %>%
    #   layout(title='Value by Country')
    
    # asinh_trans = function() trans_new("asinh", function(x) asinh(x), function(x) sinh(x))
    d$d.fi %>% select(one_of(c(group.cols, metric))) %>%
      rename_(Value=metric) %>%
      filter(!is.na(Value)) %>%
      mutate(Value=asinh(Value)) %>%
      ggplot(aes(x=Date, y=Value, color=Country)) + geom_smooth(se=F) +
      theme_bw() + ggtitle('Trends by Country')
      
  })
  
  output$fi.proj.plot <- renderPlot({
    d <- fiData()
    metrics <- input$fi.metrics
    if (is.null(metrics) || length(metrics) == 0)
      return(NULL)
    
    d <- getAssessmentFIData(d$d.fi, input$fi.assessment.id, metrics)
    if (nrow(d$proj) == 0)
      return(NULL)
    plotFIProjectDistribution(d, input$fi.plot.type)
  })
})
