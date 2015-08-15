library(SnowballC)
library(tm)
library(RWeka)
library(wordcloud)
library(RColorBrewer)
library(topicmodels)
library(dplyr)

# Load in some text data
d <- read.csv('/Users/eczech/Desktop/EmotionalDesignSurvey.csv', fileEncoding="UTF-8")

# Test data frame
#d <- data.frame(WHY=c('went running today', 'run to the store store', 'runs in the snow', 'college is long', 'college is fun', 'I went to college'), stringsAsFactors = F)

# Apply common transformations
corpus <- tm::Corpus(tm::VectorSource(d$WHY))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeWords, stopwords("english")) 
#corpus <- tm_map(corpus, stemDocument, language = "english") 

get.tdm <- function(column, stoptype=stopwords('english')){
  corpus <- Corpus(VectorSource(column))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stoptype) 
  TermDocumentMatrix(corpus)
}
BigramTokenizer <-function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
x <- get.tdm(d$WHY)
BiTDM <- TermDocumentMatrix(get.tdm(d$WHY), control = list(tokenize = BigramTokenizer))

# Create TD matrix for use in LDA
dtm <- DocumentTermMatrix(corpus)
word.counts <- colSums(as.matrix(dtm))
word.ct.sort <- sort(word.counts, decreasing = T)
as.matrix(DocumentTermMatrix(corpus))

# Determine the ideal number of topics via maximum likelihood
max.topics <- 10
topic.range <- 2:max.topics
models <- lapply(topic.range, function(d){LDA(dtm, d)}) 
model.res <- data.frame(log.lik=unlist(lapply(models, logLik)), topic.count=topic.range)
n.topics <- model.res %>% arrange(desc(log.lik)) %>% .[1,'topic.count']
#n.topics <- 10

# Run LDA using # topics above
lda <- LDA(dtm, n.topics)
topic.docs <- topics(lda)
topic.words <- terms(lda, 50)

# Create a wordcloud for each topic
for (i in 1:n.topics){
  words <- topic.words[,paste0('Topic ', i)]
  topic.cloud <- word.counts[words]
  topic.cloud <- sort(topic.cloud, decreasing = T)
  topic.cloud <- data.frame(word = names(topic.cloud),freq=topic.cloud)
  
  pal <- brewer.pal(9, "BuGn")
  pal <- pal[-(1:2)]
  png(paste0("/tmp/wordcloud_topic_", i, ".png"), width=1280,height=800)
  wordcloud(topic.cloud$word, topic.cloud$freq, 
            scale=c(8,.3), min.freq=2, max.words=100, random.order=T, rot.per=.15, 
            colors=pal, vfont=c("sans serif","plain"))
  dev.off()
}

get.docs <- function(topic){
  topic.cloud <- topic.docs[topic.docs == topic]
  d$WHY[as.integer(names(topic.cloud))]
}

lapply(1:n.topics, get.docs)



get.corpus <- function(column, stoptype=stopwords('english')){
  corpus <- Corpus(VectorSource(column))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stoptype) 
}
x <- get.corpus(d$WHY)
