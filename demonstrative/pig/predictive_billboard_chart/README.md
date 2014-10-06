Predictive Billboard Chart
==============

The [Pig script](chart_calculator.pig) in this project demonstrates the process used to create a list of artists deemed as most likely to reach the [Billboard 200](http://en.wikipedia.org/wiki/Billboard_200) at some point in their career.  This list itself is also distributed by Billboard as the [Next Big Sound Artist Chart](http://www.billboard.com/charts/next-big-sound-25).

The script utilizes a score that was already created by a supervised learning model.  This model employs stochastic gradient boosting over an ensemble of regression trees to produce likelihoods that an event will occur, and in this case, that event is appearance on the Billboard 200 ([more details on this process](http://making.nextbigsound.com/post/68287169332/predicting-next-years-breakout-artists)).  This likelihood is multiplied by 100 to produce a value from 0-100, stored in HBase, and then retrieved by this script before going through a series of filters/sorts/joins to give a final form.

Some interesting features of this script include:
 1. Joins meta data from MySQL, timeseries data from HBase, and semi-structured event data from MongoDB
 2. Incorporates HTML scraping into the pipeline itself -- something that is not easy to do in Pig
 3. Stores results in MySQL with concurrency controls -- something that is also not easy to do in Pig but can be done reliably through the ```GROUP BY (int % N) PARALLEL N``` idiom.
