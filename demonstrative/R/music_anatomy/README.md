
#Music Anatomy 201: Predicting Sales


The commercial success of music is dependent on a lot of things, but there are certainly similarities inherent to what "works" for mainstream artists.  Top country, rap, rock, and EDM acts all produce content varying in style but with certain commonalities that seem to suggest that there might be "features" of the songs themselves, independent of the artists who create them, that make them so popular.  

Electronic music is usually more uptempo with long, dramatic transitions, rock music is usually a little slower with more musical elements/instruments, country music is usually more balladic and twangy, etc.  It would be hard to argue that the potential success of new music isn't based highly on the previous success of the artist producing it, but it's worth asking if there isn't some magical formula for the components of the songs that would also be predictive in a similar way.

This analysis attempts to answer this question by comparing iTunes sales data to properties of musical content.  The properties used are provided by [EchoNest](http://the.echonest.com/) (recently [acquired by Spotify](http://techcrunch.com/2014/03/06/spotify-acquires-the-echo-nest/)) and break each song down into certain components like duration, tempo, loudness, etc.  These properties will be crossed with sales data to see if any interesting relationships jump out and whether or not they could be exploited.

Analysis Parts:
* [The Data Set](#the-data-set)
* [Analyzing Features](#analyzing-features)
* [Analyzing Sales](#analyzing-sales)
* [Analyzing Sales vs Features](#analyzing-sales-vs-features)
* [Making Predictions](#making-predictions)
* [Making Recommendations](#making-recommendations)

***
## The Data Set

To run this analysis, a data set was built containing values for the features below as well as average daily iTunes sales data for __32,310 tracks__ (by __751 artists__).  Here is a full list of all track features collected and their associated definitions:

- __time_signature__ (*integer [0,Inf)*) => An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). 
- __key__ (*integer [0,Inf)*) => The estimated overall key of a track. The key identiﬁes the tonic triad, the chord, major or minor, which represents the ﬁnal point of rest of a piece. 
- __mode__ (*binary [0,1]*) => indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.
- __tempo__ (*floating point [0,Inf)*) => The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat 
- __loudness__ (*floating point (-Inf,Inf)*) => The overall loudness of a track in decibels (dB). Loudness values in the Analyzer are averaged across an entire track and are useful for comparing relative loudness of segments and tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude)
- __duration__ (*floating point [0,Inf)*) => The duration of a track in seconds as precisely computed by the audio decoder
- __speechiness__  (*floating point [0,1]*) => Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- __liveness__  (*floating point [0,1]*) => Detects the presence of an audience in the recording. The more confident that the track is live, the closer to 1.0 the attribute value. Due to the relatively small population of live tracks in the overall domain, the threshold for detecting liveness is higher than for speechiness. A value above 0.8 provides strong likelihood that the track is live. Values between 0.6 and 0.8 describe tracks that may or may not be live or contain simulated audience sounds at the beginning or end. Values below 0.6 most likely represent studio recordings.
- __acousticness__ (*floating point [0,1]*) => Represents the likelihood a recording was created by solely acoustic means such as voice and acoustic instruments as opposed to electronically such as with synthesized, amplified, or effected instruments. Tracks with low acousticness include electric guitars, distortion, synthesizers, auto-tuned vocals, and drum machines, whereas songs with orchestral instruments, acoustic guitars, unaltered voice, and natural drum kits will have acousticness values closer to 1.0.
- __danceability__  (*floating point [0,1]*) => Represents a perceptual measure of intensity and powerful activity released throughout the track. Typical energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- __valence__ (*floating point [0,1]*) => Describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g., happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). This attribute in combination with energy is a strong indicator of acoustic mood, the general emotional qualities that may characterize the track's acoustics. Note that in the case of vocal music, lyrics may differ semantically from the perceived acoustic mood.
</ul>
</font>

After having collected the values for the features above, the result was then combined with average daily iTunes sales data (over the known lifetime of a track) to form the full data set.

Let's load it into R and take a peek:

tfa_prelim.R
```
data = tfa$get_data()

nrow(data)
# 32310

print(head(data[sample(1:nrow(data), 10),]), row.names=F)
# ** Sales numbers below are not shown directly for contractual reasons
# artist_id         artist_name                 track_name time_signature   energy liveness   tempo speechiness acousticness danceability instrumentalness key duration loudness  valence mode      sales
#      6424 Black Label Society       In This River (Live)              4 0.577994 0.399008 130.937    0.033304     0.875915     0.235197         0.390582   0 400.9995   -7.858 0.086943    1  xxx.xx
#    316132       Pistol Annies Don't Talk About Him, Tina              4 0.723416 0.250931 129.921    0.028528     0.101711     0.657455         0.000000   2 211.3329   -5.136 0.703141    1  xxx.xx
#     54121            Bon Jovi              Into The Echo              4 0.760531 0.151477 108.969    0.033131     0.002922     0.456705         0.000000   2 304.9729   -4.357 0.202757    1  xxx.xx
#     83778                Yuri                     Veneno              4 0.435456 0.074124 115.649    0.031977     0.181227     0.725501         0.032321   0 272.1839  -18.404 0.961498    0  xxx.xx
#     58235                Cher                  The Power              4 0.698579 0.058306 143.135    0.030275     0.049927     0.740907         0.000000  11 236.3596   -7.217 0.740570    0  xxx.xx
#      2959           Steve Vai                Die To Live              1 0.704459 0.989689 127.900    0.039694     0.002034     0.435893         0.585709   2 390.3734   -9.413 0.467010    1  xxx.xx
```

Hopefully most of that is intuitive.  Each record corresponds to a single track the entire set consists of only complete cases (i.e. no NA's).

***
## Analyzing Features

The sales numbers aside for the moment, lets first look at the features themselves and see what kind of data we're dealing with.  

### Feature Densities

For each feature, let's see how its values are distributed:

tfa_features.R
```
# Fetch raw data and 'stack' it, pivoting columns for features into separate rows
stacked = melt(tfa$get_data() , id.vars=c('artist_id', 'artist_name', 'track_name', 'sales'))
# > print(head(stacked[sample(1:nrow(stacked), 10),]), row.names=F)
# ** Sales numbers below are not shown directly for contractual reasons
#  artist_id   artist_name            track_name    sales    variable      value
#     300515     Olly Murs     I Blame Hollywood   xxx.xx         key   4.000000
#     116098    Trey Songz          We Should Be   xxx.xx       tempo 118.341000
#      14249 Janet Jackson 20 Part 4 (Interlude)   xxx.xx speechiness   0.034669

# Plot density by feature
ggplot(stacked, aes(x=value)) +  
  geom_histogram(aes(y=..density..), fill=NA, color='black') + 
  geom_density(color='blue') + 
  facet_wrap(~ variable, scales="free") +
  ggtitle('Track Feature Densities') + 
  xlab('Feature Value') + ylab('Density')
```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/feature_densities.png"/>

Looks interesting, but without knowing what to expect, it's hard to say if there's anything abnormal here.  We'll assume there isn't but now we at least know that the distributions of most feature values are not necessarily guassian or uni-modal, and that several only exist on a discrete scale (e.g. ``key``, ``mode``, and ``time_signature``).

### Feature Clustering

We might also expect that there should be clusters of data points where the features are "similar" based on the genre of the music.  We'd imagine that clustering all the different cases might lead to separate groups for rap, rock, dance, country, etc.  To check this, we'll try scaling all 12 feature dimensions down to 2 in the hope that some obvious clusters or trends emerge:

tfa_features.R
```
# Sample the original dataset to avoid performance issues w/ scaling operations
set.seed(123)
data_sample = data[sample(1:nrow(data), 10000),]

# Restrict to only feature columns (i.e. those not for the track/artist name or sales)
feature_data = data_sample[,tfa$get_feature_columns()]

# Scale each feature to unit variance and 0 mean
feature_data = apply(feature_data, 2, scale)

# Calculate distiances and run MDS
feature_dist = dist(feature_data)
feature_mds = as.data.frame(cmdscale(feature_dist))

# Plot the results
ggplot(feature_mds, aes(x=V1, y=V2)) + geom_point() + ggtitle('2-d Feature Value Scaling')

```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/feature_mds.png"/>

Hmm doesn't look like there's anything to this.  If there were really clusters of tracks where most of the feature values were similar, then we'd expect several regions of high density.  That's not the case though so we can rule out the possibility of there being any way to naturally segment the data based solely on the features.

### Feature Components

There are some patterns in the variations though, or in other words, it doesn't look *entirely* random.  Let's explore this some more.  Rather than looking for groups of tracks with similar features, let's try to explain the most important ways in which they vary.

The easiest way to do this is probably with PCA:

tfa_features.R
```
# Restrict to only feature columns (i.e. those not for the track name, artist, or sales)
feature_data = data[,tfa$get_feature_columns()]

# Scale each value to unit variance and 0 mean
feature_data = apply(feature_data, 2, scale)

# Run PCA and plot result
feature_pca = princomp(feature_data)

biplot(feature_pca, xlabs=rep(".", nrow(data)), main='Feature Bi-Plot', cex=1, col=c('#999999', 'red'), expand=.8)
```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/feature_pca.png"/>

As expected, there are some absurd outliers that are way different from the other tracks.  The worst of which is called '[Ik Onkar](http://grooveshark.com/#!/s/Ek+Onkar/1ams5O?src=5)' by Indian singer-songwriter [AR Rahman](http://en.wikipedia.org/wiki/A._R._Rahman).  The song itself is pretty weird, there are no instruments and it's mostly just women harmonizing over background white noise of some kind so understandably, properties like danceability, energy, and valence are off the chart in some direction.  After removing it though, we can see there are a few things worth noting.

1. There aren't any dominant directions of variance.  As expected based on how poorly the data clustered, there is no one thing that makes most tracks different.
2. While no one feature explains the differences well, groups of them do.  Energy and loudness go hand-in-hand yet danceability and valence seem like significantly different dimensions of music.  This makes sense, since Rock/Rap music would likely be common amongst tracks exhibiting more of the former while EDM/Pop tracks would likely dominate the latter.  There also seems to be a 3rd dimension for acousticness and speechiness that would likely be highest in Country/Folk songs.
3. Features like time_signature, key, and tempo don't explain much variation which also makes sense given that their values are probably pretty common in all different types of music.

Overall, there really isn't anything groundbreaking we can say after just looking at the track features, but we also didn't find any major contradictions to inuition.  This means, at the very least, that the data set is clean (i.e. the EchoNest values are legitimate)  and representative of reality so more intersting analysis of how the features relate to sales won't be based on total nonsense. 

***
## Analyzing Sales

Before we get to how features relate to sales, let's look at sales numbers in isolation:

tfa_sales.R
```
data = tfa$get_data() 

# Top level summary of sales data
summary(data$sales)

#    Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#   -1.00     1.05     1.61    60.53     6.33 43730.00 

# The distribution is skewed so check the quantiles for a better idea of how much so
data.frame(value=quantile(data$sales, probs = seq(0, 1, .1)))
#             value
# 0%      -1.000000
# 10%      0.992242
# 20%      1.012821
# 30%      1.091398
# 40%      1.250000
# 50%      1.614310
# 60%      2.354303
# 70%      4.295564
# 80%     10.377088
# 90%     41.925116
# 100% 43731.760204

```

Oof, they're not very friendly numbers and this is something we see commonly at Next Big Sound.  Engagement and sales metrics tend to be very skewed leading to extreme imbalances like this where the mean sales numbers are 3700% higher than median sales.  

It might not be entirely suprising that the majority of tracks sell less than 2 units per day, but it might be a surprise to know that selling more than just 10 units a day puts a track near the 80th percentile overall.  It's also strange to see negative numbers in there, but this is actually possible because "returning" a track on iTunes results as a negative sale (as far as we know, this is only possible via the "Complete My Album" feature) and we don't have an exhaustive history of sales data (i.e. the average can be negative because we just don't know about the original transaction).

Regardless, lets take a look at some visualizations to get a better sense of the distribution shape:
```
g = ggplot(data, aes(x=sales)) + geom_freqpoly() + xlab('Sales Value') + ylab('Frequency')

g1 = g + ggtitle('Frequency of Sales Numbers')
g2 = g + ggtitle('Frequency of Log Sales Numbers') + scale_x_log10(labels=comma)
  
do.call(grid.arrange, list(g1, g2, ncol=2))
```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/sales_frequencies.png"/>

This gives a little better sense of the shape of the distribution, and we can see that using the log of the sales numbers (in the figure on the right) scales the frequencies into something with a more discernable shape.  This transformation will be especially useful later on when looking at how features relate to sales.  

Note that the log for 0 values is non-finite and undefined for anything less than that so this visualization is slightly misrepresentative.  It turns out though that there are only 34 tracks left out for that reason so we won't sweat it right now.  In the future, we'll use an [Inverse Hyperbolic Sine](http://mathworld.wolfram.com/InverseHyperbolicSine.html) transformation (*asinh* in R) instead that behaves similarly but doesn't disregard these edge cases.


***
## Analyzing Sales vs Features

As a first cut at seeing how sales and features are related, let's first pick one feature and focus on its relationship.  Starting with the duraction of songs, let's see how sales increase or decrease based on changes in duration:

adapted from tfa_cross.R
```
# Fetch raw data and 'stack' it, pivoting columns for features into separate rows
stacked = melt(tfa$get_data() , id.vars=c('artist_id', 'artist_name', 'track_name', 'sales'))
# Data here is now the same as in the code for the "Feature Densities" section

# Function returning base ggplot object for 3-d density (to be used in later plots)
get_base_plot = function(data){
  ggplot(data, aes(x=value, y=sales)) +  
    stat_density2d(aes(alpha=..level.., fill=..level.., color=..level..), geom="polygon") +
    # Scale sales using "Inverse Hyperbolic Sine" function mentioned in the "Analyzing Sales" section
    scale_y_continuous(trans=trans_new('ihs', asinh, sinh, breaks=pretty_breaks()))
}

# Select only the data relating to duration and get a base 3-d plot for it
duration_data = subset(stacked, variable == 'duration')
graph = get_base_plot(duration_data)

# Plot sales vs duration
graph = graph +
  ggtitle('Duration vs. Sales') + ylab('Sales') + xlab('Duration') 
  
  # Add vertical lines separating regions of high sales density
  geom_vline(xintercept = c(175, 275), colour="red", linetype = "longdash") +
  
  # Create custom legend for color scale and disable legends for other aesthetics
  scale_color_continuous(name = 'Density', guide = 'colourbar', breaks=c(.001, .007), labels=c('Fewer Tracks', 'More Tracks')) +
  scale_fill_continuous(guide=FALSE) + scale_alpha_continuous(guide=FALSE) +
  
  # Add labels to vertical lines
  geom_text(aes(x=170, label="175 Seconds", y=4), colour="red", angle=90, text=element_text(size=10)) + 
  geom_text(aes(x=280, label="275 Seconds", y=4), colour="red", angle=90, text=element_text(size=10)) 
```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/cross_duration_sales.png"/>

This is a bit to take in at first, but it definitely tells an interesting story.  We see the duration of songs on the x-axis and sales on the y-axis as well as "countour regions" of different color that indicate the number of tracks that correspond to a given combination of sales and song duration.  In other words, the large majority of tracks have a low, near-zero level of sales and a duration between 3:20 (minutes:seconds) and 4:10 so this is why the color of the countour there is an intense, light blue.  On the other hand we see regions of a darker, less intense blue coloring for durations greater than 5 minutes and less than 2 and a half minutes, indicating that fewer songs have a length that is more or less than those values.

What is most promising here is that the tracks with the highest level of sales virtually all fall between 3 to 4 and a half minutes in length.  This means that if you knew nothing about a song other than how long it is, you could start to predict how successful it will be.  Most would probably say this obvious since ridiculously long or short songs aren't likely to be popular in a mainstream market, but perhaps looking at the same relationships with more sophisticated features will give us a more powerful way to predict success.

To that end, let's use the same visualization with all the other features:
\*Note that this is only for the *continuous* features -- the discrete ones like mode and time_signature don't show anything helpful

<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/cross_all.png"/>


Now we can see that a similar relationship actually exists for several features.  Liveness, loudness, danceability and several others all have ranges over which sales are maximized.  This is getting closer to something we can make predictions with but it's also pretty clear that the relationships between sales and the features aren't linear.  In other words, sales are likely to be higher as danceability increases but only to a point.  After that point, being more "danceable" makes a track less likely to sell well.  This is true for the other features too so we can't always assume that more of any one of them is necessarily a good thing.
 
Combining all of these relationships together to figure out which combinations of features is best will be tricky, but in the next section we'll skip ahead a little and just see how much predictive power there really is in this information even if the optimal feature values aren't yet known.

***
## Making Predictions

Now that we've seen that there *seems* to be a way to determine which feature values lead to better sales, we'll try predicting sales with machine learning techniques to see how well they can do.

First, let's create a problem to solve that's easy to wrap our heads around.  We'll start by simply trying to predict if a track will be a "hit" where being a hit means having sales in the top 50% of all tracks:

adapted from tfa_classify.R
```
data = tfa$get_data() 

# Remove columns for artist and track names or ids
model_data = data[,c('sales', tfa$get_feature_columns())]

# Determine what level of sales corresponds to the 50th percentile
percentile_cutoff = .5
sales_at_cutoff = quantile(model_data$sales, probs=percentile_cutoff)

# Create a variable equal to "1" when the sales value is greater than 'sales_at_cutoff' or "0" otherwise
model_data$is_hit = factor(as.numeric(model_data$sales > sales_at_cutoff))

# Predict the "is_hit" value over 10 folds of the data set and plot performance measures
k = 10
results = tfa_classify$get_modeling_results(model_data, k)
tfa_classify$plot_modeling_results(results, k)
```
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/classifier_perf_no_tracks.png"/>

The underlying random forest model applied here produces probabilities, like most binary classifiers, of a track being a hit.  In the first figure we see how accurate the classifier is based on the cutoff for the output probabilities, above and below which we would predict that a track is a hit.  Not surprisingly, the best cutoff is around .5 and if we use that value the accuracy overall is around __63%__.

In the second figure, we see how often each kind of classification occurs and it's pretty clear that the number of "true" or accurate hit/no hit predictions is a good bit higher than the number of "false" or incorrect ones.  Statistically speaking this result is definitely significant -- if we just flipped a coin instead to make predictions the chance of getting 63% of them correct for thousands of tracks is basically 0.  Practically speaking though, the usefulness of this accuracy is debatable.

We can improve on this by adding some more information to the model that we already have.  If we simply add the *number* of tracks an artist has ever sold on iTunes to the model we can get a small performance boost:

adapted from tfa_classify.R
```
# Fetch the raw data but also include artist and track name this time
extra_cols = c('artist_id', 'track_name')
model_data = data[,c('sales', tfa$get_feature_columns(), extra_cols)]

# Compute the number of tracks per artist and remove non-predictor fields
model_data = ddply(model_data, .(artist_id), transform, num_tracks=length(unique(track_name)))
model_data = model_data[, names(model_data)[!names(model_data) %in% extra_cols]]

# Again, run the model for 10 folds and plot the results
results = tfa_classify$run_model(model_data)
tfa_classify$plot_modeling_results(results, k)
```

<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/classifier_perf_tracks.png"/>

The results are pretty similar but the performance has increased to about __66%__ and the ``num_tracks`` variable is now the most important predictor.  A 3% boost might not seem like much but [McNemar's Test](http://en.wikipedia.org/wiki/McNemar's_test) shows that it's definitely a statistically significant result (p < .001).

We could add other variables too like the artist, and having tried that it does turn out to be a more powerful predictor than anything else.  Using the artist can boost performance to over 70% but it's kind of cheating since the point of the analysis is to see what can be learned from only the tracks.

***
## Making Recommendations

It turns out that there are some solid recommendations that can be made about how the properties of music affect sales.  Putting everything together, we can make generalizations about how to optimize content based on the most important predictors to higher levels of sales:

\* Note that this tree is based on the first predictive model that did NOT include track counts
<img src="https://dl.dropboxusercontent.com/u/65158725/tfa/decision_tree.png"/>

This decision tree tells us whether or not a track will sell in the top 50th percentile based on the most predictive properties we've discovered so far (all other properties not shown just don't make much of a difference).  The criterion at each node determines which way the track falls in the tree and once a "leaf" node is reached, the 0 or 1 value indicates the prediction (0 = bottom 50%, 1 = top 50%).

We can gather several things from this, namely:

1. If you want your track to sell well, don't make it too short.  Based on this tree and what we saw in the [Analyzing Sales vs Features](#analyzing-sales-vs-features) section, songs that are less than ~2.5 minutes do poorly and this is the single most important predictor of success.  To speculate, we'd guess this is because nobody wants to pay the same price for less content -- which is hard to argue.

2. Make your music "loud".  The "[loudness wars](http://musicmachinery.com/2009/03/23/the-loudness-war/)" as it's been dubbed by Paul Lamere of [Music Machinery](http://musicmachinery.com/) is a recent trend where music is engineered to catch listeners' attention early, and it works!  Having a *loudness* of greater than -6 on the EchoNest dB scale will give any track a better shot at getting noticed.

3. If your track isn't engineered in some lab to be as loud and as catchy as possible, then the next best way to go is to make it neither too "acoustic" nor too "happy".  In other words, stick to electronic instruments with a more mellow vibe (think Pheonix, Arcade Fire, Weezer, etc.).  If a song has an *acousticness* level that is too high or failing that, a *valence* (i.e. happiness) level too high, it probably won't do as well.  There are certainly exceptions, but here are some examples to make the distinction more clear:

| Ideal (not too happy or acoustic) | Too acoustic | Less acoustic but too happy |
------------------------------------|--------------|------------------------------
| Locked Out Of Heaven - Bruno Mars | Pocket Philosopher - Mandy Moore | Dig Your Own Hole - Gotye |
| Sweet Nothing - Calvin Harris | Keep Warm - Ingrid Michaelson | Brand New Luv - Robin Thicke |
| Take A Walk - Passion Pit | Inside Out - Sara Bareilles | Invisible - Will.I.Am |
| We Exist - Arcade Fire | Outside - The Weeknd | Blame It On Me - Barenaked Ladies |

The boundaries for the best feature values prescribed by the decision tree above are helpful, but they make the guidelines a little harder to follow than they should be.  Based on the sales densities shown in [Analyzing Sales vs Features](#analyzing-sales-vs-features) we might expect that we should be able to get ideal __ranges__ for each feature value and not just single less than or greater than criterion like those in the tree.  The tree does this because the feature values tend to vary together, as shown in [Analyzing Features](#analyzing-features), so rather than recommending to keep a single feature value in a range it would say to keep the value of one feature high and another feature low because it accomplishes roughly the same thing.  

A more straightforward and generalized set of guidelines to follow would be to create tracks with content falling in these ranges: 

| Feature Name | Ideal Range |
---------------|-------------
duration  | 175 to 275 seconds
speechiness | .025 to .05
acousticness | 0 to .1
danceability | .35 to .75
loudness | -9 to -1
valence | 0 to .75







