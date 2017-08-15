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


setwd("/Users/seanmhendryx/reach_context-balancing/reach/")
DT = as.data.table(read_feather("features.feather"))

p = ggplot(data = DT, mapping = aes(x = min_sentenceDistance, group = label, fill = label)) + geom_histogram(position = "dodge", binwidth = .25) + theme_bw()

p
