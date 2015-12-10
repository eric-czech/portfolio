results_1
========================================================
author: 
date: 


Plots
========================================================

![plot of chunk unnamed-chunk-1](results_1-figure/unnamed-chunk-1-1.png) 

Nonlinear Modeling 
========================================================
A modified model:

$$ logit(y_i) = \alpha + \beta \cdot X_i + f(G_{ij}) $$

where

$$ X_i = [Gender_i, Age_i, CommaScore_i, MarshallScore_i] $$

and

$$ f(G_i) = \frac{1}{n_i} \sum_j{ \frac{c_1}{1 + e^{-c_2(G_{ij} - c_3)}} + \frac{c_4}{1 + e^{-c_5(G_{ij} - c_6)}}  } $$
$$ n_i = \text{ length of timeseries for patient }i $$

Double Logistic Function Examples
========================================================

![plot of chunk unnamed-chunk-2](results_1-figure/unnamed-chunk-2-1.png) 

Random Functions
========================================================

These are functions drawn from the priors in the model and show all possibilities:

<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_prior.png"/>

Nonlinear Effects
========================================================
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/presentations/images/NonlinearEffect.png"/>

Sample Size Effects (Fully Simulated Data)
========================================================

<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_800.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_600.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_400.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_250.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_100.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/sim_1_50.png" width="300px" height="300px"/>


Function Forms on Semi-Simulated Data
========================================================
By semi-simulated, I mean by taking the real data and hard coding coefficient / function values.

<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/act_upward.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/act_slope.png" width="300px" height="300px"/>
<img src="/Users/eczech/repos/portfolio/demonstrative/R/pbto2/sim/images/act_downward.png" width="300px" height="300px"/>

GLM Model Results
========================================================

<!--Results from a few ordinal GLMs -->



Exhaustive Model Results
========================================================

Results from exhaustive binary model search


Cutpoint Model Results
========================================================

Results from cutpoint modeling


Nonlinear Models
========================================================




```r
summary(cars)
```

```
     speed           dist       
 Min.   : 4.0   Min.   :  2.00  
 1st Qu.:12.0   1st Qu.: 26.00  
 Median :15.0   Median : 36.00  
 Mean   :15.4   Mean   : 42.98  
 3rd Qu.:19.0   3rd Qu.: 56.00  
 Max.   :25.0   Max.   :120.00  
```

Slide With Plot
========================================================

![plot of chunk unnamed-chunk-4](results_1-figure/unnamed-chunk-4-1.png) 
