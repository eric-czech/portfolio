
shinyUI(pageWithSidebar(
  headerPanel('Event Scoring'),
  sidebarPanel(
    selectInput('data.type', 'Simulated Event Type', c('Fixed', 'Variable')),
    numericInput('prior.alpha', 'Prior Alpha', 1),
    numericInput('prior.beta', 'Prior Beta', 10),
    #checkboxInput('enable.smoothing', 'Enable Smoothing', T),
    numericInput('smooth.parameter', 'Smooth Parameter [0-1]', value=.9, min=0, max=1, step=.1),
    numericInput('window.size', 'Window Size in Days [1,]', value=30, min=1, step=1),
    numericInput('credible.interval.p', 'Credible Interval Percentile [0-1]', value=.025, min=0, max=1, step=.025),
    actionButton("plot.button", "Plot"),
    width=3
  ),
  mainPanel(
    plotOutput('plot.prior', height="150px"),
    plotOutput('plot.events', height="200px"),
    plotOutput('plot.estimates', height="200px"),
    plotOutput('plot.scores', height="200px")
  )
))