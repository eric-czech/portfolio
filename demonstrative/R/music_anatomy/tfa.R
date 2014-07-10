library(logging)

# Initialize logging
basicConfig()

# Data definition and retrieval envrionment for Track Feature Analysis (TFA) project
tfa = new.env()

tfa$get_data = function(file='/home/nbs/data/echonest/track_feature_vs_sales.csv'){
  #' Loads TFA data set contained in the given file.
  #' 
  #' Args:
  #'   file: location of TFA csv
  #' Returns:
  #'   data frame
  #' Example:
  #'   data = tfa$get_data()
  #'   head(data[sample(1:nrow(data), 10),])
  #    * Actual sales numbers ommitted for contractual reasons
  # artist_id     artist_name             track_name time_signature   energy liveness   tempo speechiness acousticness danceability instrumentalness key duration loudness  valence mode     sales
  # 2094     116468   Mary J. Blige  I Found My Everything              3 0.502346 0.471188 117.383    0.043997     0.333092     0.485153          2.0e-06   3 323.0662   -4.195 0.415137    1 xxx.xx
  # 20702     35197     Rod Stewart           Father & Son              4 0.580767 0.127736 132.007    0.028759     0.444916     0.578946          0.0e+00   6 216.8129   -7.793 0.425361    1 xxx.xx
  # 16182    301919         Damares          Pode ser hoje              4 0.806038 0.128372 146.087    0.062481     0.442882     0.501898          0.0e+00   3 340.0795   -4.324 0.532895    1 xxx.xx
  # 12756    283440           Arisa Meraviglioso amore mio              4 0.277853 0.101733 129.839    0.038297     0.748558     0.659779          5.3e-05  10 236.9329   -9.297 0.172371    1 xxx.xx
  # 6730     178755         Kalimba       No Volveras A Mi              4 0.562545 0.128101 134.028    0.031002     0.226804     0.724803          7.7e-05   7 208.0929   -6.245 0.839148    1 xxx.xx
  # 17278    307857 Anjelah Johnson                  Valet              3 0.920493 0.818041  79.332    0.939402     0.814681     0.573001          1.0e-06   1 104.3862  -11.740 0.081204    0 xxx.xx
  
  loginfo(paste0('Loading raw data set from file ', file, ' ...'))
  read.csv(file=file, header=T, encoding = "utf-8", stringsAsFactors=F)
}

tfa$get_feature_columns = function(){
  #' Returns a vector of track 'Feature' column names.
  #' 
  #' These values were collected from the EchoNest API using this endpoint: http://developer.echonest.com/docs/v4/track.html
  #' 
  #' More thorough documentation on most of the fields can be found here: 
  #'  - Docs on more esoteric attributes: http://developer.echonest.com/acoustic-attributes.html
  #'  - Docs on more common attributes:   http://developer.echonest.com/docs/v4/_static/AnalyzeDocumentation.pdf
  c('time_signature','energy','liveness','tempo','speechiness','acousticness','danceability','key','duration','loudness','valence','mode')
}
