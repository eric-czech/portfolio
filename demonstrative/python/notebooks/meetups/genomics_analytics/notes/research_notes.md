---
title: "Research Notes"
author: "Eric Czech"
date: "December 18, 2015"
output: html_document
---

# [Oncotype DX](https://en.wikipedia.org/wiki/Oncotype_DX)


---------

### [Overview of Oncotype DX](http://www.medscape.com/viewarticle/779859_2)

- "The selection of the 16 cancer-related genes used for the Oncotype DX assay was based on data demonstrating a consistent and strong statistical link between the level of expression of these genes and distant breast cancer recurrence in an analysis of 250 candidate genes in a total of 447 patients from three independent clinical studies" (this is a reference from article below)
- "In multivariate Cox models that included RS, patient age, T-size and grade, as well as ER, PR or DNA amplification of HER2, only the RS and grade were found to be significant predictors of distant recurrence."

---------

### [Basis for Oncotype](http://www.nejm.org/doi/full/10.1056/NEJMoa041588#t=articleMethods)

- This paper was mentioned by above study as source of genes used in Oncotype test
- "The list of 21 genes and the recurrence-score algorithm were designed by analyzing the results of the three independent preliminary studies involving 447 patients and 250 candidate genes"
- "we analyzed data from three independent clinical studies of breast cancer involving a total of 447 patients, including the tamoxifen-only group of NSABP trial B-20, to test the relation between expression of the 250 candidate genes and the recurrence of breast cancer.24-26 Fourth, we used the results of the three studies to select a panel of 16 cancer-related genes and 5 reference genes and designed an algorithm, based on the levels of expression of these genes, to compute a recurrence score for each tumor sample"

---------


# [Magee Equations](http://path.upmc.edu/onlineTools/mageeequations.html)

----------

### [Prediction of the Oncotype DX recurrence score](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3647116/)

- Estimating Oncotype DX score by using "standard histologic variables" (Nottingham score, tumor size, H-scores) in combination with "semiquantitative ER, PR, HER2, and Ki-67 results".  These 4 groups of genes are a part of the actual Oncotype DX assay (there are other gene groups in it too though).
- The point was to instead of using fully quantative gene measurements, use ER and PR, and HER2 "status" (negative, equivocal, or positive).  Presumably this is cheaper and easier.
- Linear regressions were then done depending on what data is available resulting in 3 "Magee" equations like:
  Recurrence score=15.31385+Nottingham score*1.4055+ERIHC*(−0.01924)+PRIHC*(−0.02925)+(0 for HER2 negative, 0.77681 for equivocal, 11.58134 for HER2 positive)+tumor size*0.78677+Ki-67 index*0.13269.
- H-score explanation: "At our institution, ER and PR results are reported using a semiquantitative immunohistochemical score (commonly known as ‘H-score'), which details the percentage of positive cells showing none, weak, moderate, or strong staining.7, 8 The score is given as the sum of the percent staining multiplied by an ordinal value corresponding to the intensity level (0: none; 1+: weak; 2: moderate; 3+: strong). The resulting score ranges from 0 (no staining in the tumor) to 300 (diffuse intense staining)"
- Performance assed using confusion matrix of continuous scale quantizations and pearson correlation between predicted recurrence score and actual
- Plot of predicted scores vs actual: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3647116/figure/fig1/

#### Outside Opinions

- [Forum 1](http://forums.studentdoctor.net/threads/anyone-reporting-magee-equations-for-breast-cases.1059448/) - Someone on whether or not they would use the Magee equations:

    "No; not sure why the oncologists can't or wouldn't just do such scoring themselves...they don't need a pathologist to translate ER/PR/HER2 positivity or negativity into some prediction. That's ridiculous.  I think that goes way beyond what is necessary for diagnostic purposes (in terms of what I'm willing to spend my time doing anyway...which is not calculating "recurrence score=13.424+5.420*(nuclear grade)+ 5.538*(mitotic count)0.045*(ER H-score)0.030*(PR H-score)+9.486*(0 for negative/equivocal and1 for HER2 positive)...Jesus."

#### Relevant Questions
- Are these "histological" variables (Nottingham score, tumor size, something like H-score) available in cBioPortal and COSMIC?


-------

### [Modified Magee Equations](http://www.ncbi.nlm.nih.gov/pubmed/25932962)

- The current list price of Oncotype DX is $4,175.00.
- Histological variables can approximate this score instead
- Predicted scores used to rule out very high and very low risk cases
- Pearson correlation again used to assess predictions vs actual (.66)
- " Using an algorithmic approach to eliminate high and low risk cases, between 5% and 23% of cases would potentially not have been sent by our institution for Oncotype DX testing, creating a potential cost savings between $56,550.00 and $282,750.00"



