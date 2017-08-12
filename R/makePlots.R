library(feather)
library(data.table)
library(ggplot2)


setwd("/Users/seanmhendryx/reach_context-balancing/reach/")
DT = as.data.table(read_feather("features.feather"))

p = ggplot(data = DT, mapping = aes(x = min_sentenceDistance, group = label, fill = label)) + geom_histogram(position = "dodge", binwidth = .25) + theme_bw()

p
