palette(c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
          "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"))

source('app.R')

#runApp('/Users/eczech/repos/portfolio/demonstrative/R/bayesian_scoring/app')

shinyServer(function(input, output, session) {
  
  # Combine the selected variables into a new data frame
  selectedData <- reactive({
    sim <- new.env()
    if (input$data.type == 'Fixed')
      load('../sim/sim_data_fixed.RData', envir = sim)
    else if (input$data.type == 'Variable')
      load('../sim/sim_data_variable.RData', envir = sim)
    else
      stop(paste0('Data type "', input$data.type, '" not valid.'))
    sim$data
  })
  
  posteriorData <- reactive({
    input$plot.button
    isolate(getPosterior(selectedData(), input))
  })

  output$plot.prior <- renderPlot({
    input$plot.button
    isolate(getPriorPlot(input) )
  })
  
  output$plot.events <- renderPlot({
    input$plot.button
    isolate(getEventsPlot(posteriorData(), input) )
  })
  
  output$plot.estimates <- renderPlot({
    input$plot.button
    isolate(getEstimatesPlot(posteriorData(), input) )
  })
  
  output$plot.scores <- renderPlot({
    input$plot.button
    isolate(getScoresPlot(posteriorData(), input) )
  })
  
})