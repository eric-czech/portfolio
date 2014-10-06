Twitter Affinity Project
====================

The Pig script in this project demonstrates the process for calculating "affinity" between two entities on Twitter.  At Next Big Sound, commercial applications of this metric utilized the following definition:

Given an entity pair on Twitter with members *U<sub>1</sub>* and *U<sub>2</sub>*, "Affinity" equals the sum of the number of followers of the unique users that mention/retweet BOTH *U<sub>1</sub>* and *U<sub>2</sub>*.

This metric is meant to roughly estimate the size of the audience witness to discussions about two entities that are often discussed together.  Presumably, the higher this score is for any user pair the "stronger" the relationship
between the two is.  Or at the very least, a higher score indicates that the same people like to talk about both, regardless of what the relationship between them actually is.
