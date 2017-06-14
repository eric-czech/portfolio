Examples and Links
=======

Below is a list of recent, public projects or talks I have been a part of:

## Analysis


- **Genomics of Drug Sensitivity in Cancer** [[notebook](https://cdn.rawgit.com/eric-czech/mgds/9efe3ea8/python/notebook/data_modeling/kl_modeling/transfer_kernel_project.html)] - Summary of results for predictive models applied to genomic data as a means of determining patient sensitivity to a large number of theraputic cancer drugs.  The primary challenge in this research was not only to predict sensitivity well but to do so with relatively concise explanations for the rationale behind those predictions.  In this case, this was acheived through the use of a Bayesian Transfer Learning model (built using Tensorflow and [Edward](http://edwardlib.org/)).

- **Deep Learning** [[slides](https://cdn.rawgit.com/CharlestonDataScience/PythonNotebooks/515f5e7a/notebooks/deep_learning_01/deep_learning_presentation.slides.html) | [notebook](https://github.com/CharlestonDataScience/PythonNotebooks/blob/master/notebooks/deep_learning_01/deep_learning_presentation.ipynb)] - A [Charleston Data Analytics](http://www.meetup.com/Charleston-Data-Analytics/) talk covering neural networks, theory on how depth in networks affect expressiveness, gradient descent and Tensorflow.  Also includes [examples](https://cdn.rawgit.com/eric-czech/portfolio/3d05c545/demonstrative/python/notebooks/meetups/python_tutorials/deep_learning/pt1/tensorflow_examples.html) of how to build and apply custom Tensorflow models to clinical research data (Alzheimer's Disease in this case).

- **Bayesian Analysis** [[slides](https://cdn.rawgit.com/eric-czech/portfolio/master/demonstrative/python/notebooks/meetups/data_analysis_examples/meetup_pres.slides.html) | [notebook](https://github.com/eric-czech/portfolio/blob/master/demonstrative/python/notebooks/meetups/data_analysis_examples/meetup_pres.ipynb)] - Covers several ideas in Bayesian Modeling and Reasoning like:
  1. Bayesian ranking and modeling approaches for smaller data sets (examples in Python and Stan)
  2. Hierarchical Maximum Likelihood modeling within the context of forecasting crime rates for various Carribean countries
  3. Creating a paint-by-numbers from a digital image through the use of nonparametric, Bayesian clustering algorithms (i.e. [Dirichlet Process](https://en.wikipedia.org/wiki/Dirichlet_process))

-  **Predicting Sales Through Music Anatomy** [[project](/demonstrative/R/music_anatomy/README.md)] - Analyzing the relationship between iTunes sales and traits of music like tempo, loudness, danceability, acousticness, and more ([Forbes.com Article](http://www.forbes.com/sites/livbuli/2014/09/18/engineering-success-the-data-driven-approach-to-hit-making/)).


## Engineering

-  [HBlocks (Java, Pig, Oozie, Bash, HDFS, MySQL)](http://bit.ly/1QkU3Xt) - White paper on production storage system at Next Big Sound that spans multiple Hadoop subsystems to create a large scale (many terabyte) data revision control platform.  *No code uploaded yet, just the paper for now.*

-  [High Performance Transformations for 10M+ Record Impala Result Sets (R)](/demonstrative/R/impala/transforms.R) - data.table optimizations applied to common transformations on large data frames in R.  This was helpful at Next Big Sound for processing huge data frames when base R or plyr functions wouldn't cut it.

-  [Next Big Sound Chart Calculator (Pig)](/demonstrative/pig/predictive_billboard_chart) - Computes a list of artists most likely to appear on the [Billboard 200](http://en.wikipedia.org/wiki/Billboard_200) using likelihoods produced by a particular [supervised learning technique](http://making.nextbigsound.com/post/68287169332/predicting-next-years-breakout-artists) and stored in HBase, semi-structured event data from MongoDB, and artist meta data from MySQL.

-  [HDFS Disaster Recovery (Bash)](/demonstrative/bash/hdfs_backup/hdfs_backup.sh) - Shell script used to backup critical HDFS paths into rolling "archive" directories for offsite delivery or immediate DR.


