Data Science Portfolio
=======

Hello!  I'm [Eric](https://www.linkedin.com/pub/eric-czech/8/992/202).  

As a software engineer turned data enthusiast, I've been lucky enough to work on some pretty cool projects in the last couple years and this repo contains examples of that work.  The examples within are meant to showcase the use of modern open-source tools, core computer science concepts, and systems design principles for managing/analyzing information in a distributed, polyglottic environment (they don't really do all that yet, but hopefully they will when finished!).

All of the projects contained are broken roughly into two categories, [functional](/functional) and [demonstrative](/demonstrative).  The former includes smaller, executable codebases illustrating solutions to common problems in several different programming languages while the latter includes real-world, production (but not executable) solutions used, mostly within the context of Hadoop, to process large quantities of data at [Next Big Sound](https://www.nextbigsound.com/about).

Project Links
------------------
####Data Analysis
-  [Predicting Sales Through Music Anatomy (R)](/demonstrative/R/music_anatomy/README.md) - Analyzing the relationship between iTunes sales and traits of music like tempo, loudness, danceability, acousticness, and more ([Forbes.com Article](http://www.forbes.com/sites/livbuli/2014/09/18/engineering-success-the-data-driven-approach-to-hit-making/)).

-  [Predicting the Chemical Composition of Soil in Africa (Python)](http://nbviewer.ipython.org/github/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/kaggle_soil.ipynb) - Kaggle competition submission analyzing 3,595 different properties of soil samples in an effort to predict other properties of that same soil like pH, sand content, and Phosphorous/Carbon/Calcium levels.

-  [Phone Bill Classification (Python)](http://nbviewer.ipython.org/github/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/phone_bills.ipynb) - Expensing business calls on my Verizon Wireless bill using an ensemble tree classifier.

####Data Engineering
-  [Calculating "Affinity" Between Entities on Twitter (Pig)](/demonstrative/pig/twitter_affinity/twitter_affinity.pig) - Determining the strentgh of pairwise relationships between Twitter users by analyzing the reach of the users that engage with both entities in any given pair -- this was a valuable tool at Next Big Sound for recommending artist sponsorships to various brands. 
-  [High Performance Transformations for 10M+ Record Impala Result Sets (R)](/demonstrative/R/impala/transforms.R) - data.table optimizations applied to common transformations on large data frames in R.  This was helpful at Next Big Sound for processing huge data frames when base R or plyr functions wouldn't cut it.

####Data Systems
-  [HBlocks (Java, Pig, Oozie, Bash, HDFS, MySQL)](http://bit.ly/1rCkZJS) - White paper on production storage system at Next Big Sound that spans multiple Hadoop subsystems to create a large scale (many terabyte) data revision control platform.  *No code uploaded yet, just the paper for now.*
-  [HDFS Disaster Recovery (Bash)](/demonstrative/bash/hdfs_backup/hdfs_backup.sh) - Shell script used to backup critical HDFS paths into rolling "archive" directories for offsite delivery or immediate DR.


