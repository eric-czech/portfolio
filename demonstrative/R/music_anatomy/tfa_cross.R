source('tfa.R')

library(ggplot2)
library(reshape2)
library(scales)
library(plyr)

# Envrionment for common functions
tfa_cross = new.env()

# Fetch raw data and stack it
data = tfa$get_data() 
stacked = melt(data, id.vars=c('artist_id', 'artist_name', 'track_name', 'sales'))

# Base 2-d density plotting function
tfa_cross$get_base_plot = function(data){
  ggplot(data, aes(x=value, y=sales)) +  
    stat_density2d(aes(alpha=..level.., fill=..level.., color=..level..), geom="polygon") + 
    scale_y_continuous(trans=trans_new('ihs', asinh, sinh, breaks=pretty_breaks()))
}

# Plot sales vs duration
tfa_cross$get_base_plot(subset(stacked, variable == 'duration')) +
  ggtitle('Duration vs. Sales') + ylab('Sales') + xlab('Duration') + 
  
  # Add vertical lines separating regions of higher sales density
  geom_vline(xintercept = c(175, 275), colour="red", linetype = "longdash") +
  
  # Create custom legend for color scale and disable legends for other aesthetics
  scale_color_continuous(name = 'Density', guide = 'colourbar', breaks=c(.001, .007), labels=c('Fewer Tracks', 'More Tracks')) +
  scale_fill_continuous(guide=FALSE) + scale_alpha_continuous(guide=FALSE) +
  
  # Add labels to vertical lines
  geom_text(aes(x=170, label="175 Seconds", y=4), colour="red", angle=90, text=element_text(size=10)) + 
  geom_text(aes(x=280, label="275 Seconds", y=4), colour="red", angle=90, text=element_text(size=10)) 


# Plot sales vs ALL continous features
graphs = list()
d_ply(stacked, .(variable), function(data){
  # Ignore discrete variables
  if (data$variable[1] %in% c('time_signature', 'key', 'mode')){
    return(NA)
  }
  g = tfa_cross$get_base_plot(data) +
    ggtitle(data$variable[1]) +
    # Disable all legends
    theme(legend.position="none", axis.title.x=element_blank(), axis.title.y=element_blank())
  graphs <<- c(graphs, list(g))
})
graphs$main = 'Feature vs Sales Density'
graphs$left = 'Sales'
graphs$sub = 'Feature Value'
do.call(grid.arrange, graphs)

