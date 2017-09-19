# File runs cross validation using sklearn functions
#
# Authored by Sean M. Hendryx while working at the University of Arizona
# contact: seanmhendryx@email.arizona.edu https://github.com/SMHendryx/binaryClassifierParameterGridSearch
# Copyright (c)  2017 Sean Hendryx
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################################################################################################


library(feather)
library(data.table)
library(ggplot2)
library(plotly)


setwd("/Users/seanmhendryx/reach_context-balancing/reach/")
DT = as.data.table(read_feather("features.feather"))

#Compare papers by Euclidean distance to feature centroid:
F = DT[,5:ncol(DT)]
distances = DT[,1:4]
distances[,distance := numeric()]

centroid = colMeans(F)

# Vectorized:
diff = t(t(F) - centroid)
#distances[,centroidDiff := centroid - distances]
squared = diff^2
sumSquaredDiff = rowSums(squared)
EuclideanDistance = sqrt(sumSquaredDiff)
rm(diff)
rm(sumSquaredDiff)
distances[,distance := EuclideanDistance]

#for(i in seq(nrow(F))){
#  distances[i,distance := sqrt(sum((centroid - F[i,])^2))]
#}

# Now compute mean and standard deviation of distance by paper:
papers = unique(DT[,PMCID])



for(paper in papers){
  distances[PMCID == paper , meanPaperDistance := mean(distance)]
  distances[PMCID == paper , sdPaperDistance := sd(distance)]
}

#plot mean and sd of distance from feature centroid to show potential clusterings
p = ggplot(data = distances, mapping = aes(x = meanPaperDistance, y = sdPaperDistance, fill = PMCID)) + geom_point() + theme_bw()
p
ply = ggplotly(p)
ply

# TODO:
# Make density-histogram of mean distances:

# TODO:
# Host

p = ggplot(data = DT, mapping = aes(x = min_sentenceDistance, group = label, fill = label)) + geom_histogram(position = "dodge", binwidth = .25) + theme_bw()

p
