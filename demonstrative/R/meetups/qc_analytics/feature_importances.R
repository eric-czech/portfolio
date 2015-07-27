library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(foreach)

feats <- read.csv('/tmp/feats.csv')

feats <- feats %>% 
  mutate(Value = abs(as.numeric(feats$Value))) %>%
  group_by(Algorithm) %>% do({ arrange(., desc(Value))[1:10,] })


grobs <- foreach(algorithm=unique(feats$Algorithm)) %do% {
  g <- feats %>%
    filter(Algorithm == algorithm)
  ofeat <- g %>% arrange(desc(Value)) %>% .$Feature
  g$Feature = factor(as.character(g$Feature), levels=ofeat)
  
  g <- ggplot(g, aes(x=Feature, y=Value)) + geom_bar(stat='identity') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(algorithm)
  ggplotGrob(g)
}
grid.newpage()
grid.arrange(grobs[[1]], grobs[[2]], grobs[[3]], grobs[[4]], grobs[[5]], grobs[[5]], nrow=3, ncol=2) 
