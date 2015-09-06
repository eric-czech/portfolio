Data Science Portfolio
=======

This repository contains examples of some the projects I've had the pleasure to work on over the last few years.  The guide below is meant to demonstrate a small but representative portion of the details of those projects, spanning a spectrum that involves at least some of the following skills:

- __Statistical Research__ - IPython and R notebooks and slides
- __Application Development__ - Large-scale database systems built on top of Hadoop (Hortonworks and Cloudera), Java, Pig, Hive, and MySQL
- __Systems Administration__ - Linux environment management, Bash scripting, and Chef cookbooks

Project Links
------------------

####Data Analysis

- [Bayesian Analysis Presentation](https://github.com/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/meetups/data_analysis_examples/meetup_pres.ipynb) - Slides from a talk given to the [Charleston Data Analytics](http://www.meetup.com/Charleston-Data-Analytics/) meetup group about the following:
  1. Bayesian ranking and modeling approaches for smaller data sets (examples in Python and Stan)
  2. Hierarchical Maximum Likelihood modeling within the context of forecasting crime rates for various Carribean countries
  3. Creating a paint-by-numbers from a digitical image through the use of nonparametric, Bayesian clustering algorithms (i.e. [Dirichlet Process](https://en.wikipedia.org/wiki/Dirichlet_process))

-  [Predicting Sales Through Music Anatomy (R)](/demonstrative/R/music_anatomy/README.md) - Analyzing the relationship between iTunes sales and traits of music like tempo, loudness, danceability, acousticness, and more ([Forbes.com Article](http://www.forbes.com/sites/livbuli/2014/09/18/engineering-success-the-data-driven-approach-to-hit-making/)).

-  [Predicting the Chemical Composition of Soil in Africa (Python)](http://nbviewer.ipython.org/github/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/kaggle/kaggle_soil.ipynb) - Kaggle competition submission analyzing 3,595 different properties of soil samples in an effort to predict other properties of that same soil like pH, sand content, and Phosphorous/Carbon/Calcium levels.

-  [Phone Bill Classification (Python)](http://nbviewer.ipython.org/github/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/phone_bills.ipynb) - Expensing business calls on my Verizon Wireless bill using an ensemble tree classifier.

-  [Calculating "Affinity" Between Entities on Twitter (Pig)](/demonstrative/pig/twitter_affinity) - Determining the strentgh of pairwise relationships between Twitter users by analyzing the reach of the users that engage with both entities in any given pair -- this was a valuable tool at Next Big Sound for recommending artist sponsorships to various brands. 


####Data Engineering
-  [High Performance Transformations for 10M+ Record Impala Result Sets (R)](/demonstrative/R/impala/transforms.R) - data.table optimizations applied to common transformations on large data frames in R.  This was helpful at Next Big Sound for processing huge data frames when base R or plyr functions wouldn't cut it.

-  [Next Big Sound Chart Calculator (Pig)](/demonstrative/pig/predictive_billboard_chart) - Computes a list of artists most likely to appear on the [Billboard 200](http://en.wikipedia.org/wiki/Billboard_200) using likelihoods produced by a particular [supervised learning technique](http://making.nextbigsound.com/post/68287169332/predicting-next-years-breakout-artists) and stored in HBase, semi-structured event data from MongoDB, and artist meta data from MySQL.


####Data Systems
-  [HBlocks (Java, Pig, Oozie, Bash, HDFS, MySQL)](http://bit.ly/1rCkZJS) - White paper on production storage system at Next Big Sound that spans multiple Hadoop subsystems to create a large scale (many terabyte) data revision control platform.  *No code uploaded yet, just the paper for now.*

-  [HDFS Disaster Recovery (Bash)](/demonstrative/bash/hdfs_backup/hdfs_backup.sh) - Shell script used to backup critical HDFS paths into rolling "archive" directories for offsite delivery or immediate DR.


