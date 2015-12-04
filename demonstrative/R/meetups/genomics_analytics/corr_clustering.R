library(dplyr)
library(ggplot2)
library(gplots)

#f <- '/Users/eczech/repos/misc/Charleston-Data-Analytics-Cancer-Genomics-Data-Challenge/modeling/data/data_n_ho_30_training.csv.gz'
f <- '/tmp/d_tr.csv'
#d <- read.table(gzfile(f), sep=',', header = T)
d <- read.table(f, sep=',', header = T)

dt <- d %>% select(-tumorID, -res_AUC, -res_IC50, -type)

cors <- cor(dt, method='pearson')

heatmap.2(cors, main="Hierarchical Cluster", dendrogram="column", trace="none", col=rich.colors(10))