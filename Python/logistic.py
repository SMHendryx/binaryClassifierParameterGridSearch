# File runs cross validation on binary logistic regression classifier using sklearn functions
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


import sys
import os
import pandas
import feather

import numpy as np
#import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics


# read in data:
dir = '/Users/seanmhendryx/Data/context'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)
print("Data successfully read in.")

print("Training models and running CV with only sentenceDistance feature:")
# set up data:
#sentenceDistance is the distance between the event and context mention.
X = df['sentenceDistance']
y = df['label']
X = X.reshape((X.size, 1))
y = y.reshape((y.size, 1))

# instantiate logistic regression object
LR = LogisticRegression(penalty='l1')

#fit 
#model = LR.fit(X, y)

#VALIDATE
#Apparently cross_val_score  requires an (R,) shape label instead of (R,1) (which is confusing bc LogisticRegression requires the opposite):
c, r = y.shape
del r
y = y.reshape(c,)
#scores = cross_val_score(LR, X, y, cv=10, scoring='f1_micro')
#np.mean(scores)

#as opposed to cross_val_score, cross_validate returns a dict of float arrays of shape=(n_splits,), including 'test_score', 'train_score', 'fit_time', and 'score_time'
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')

print(cvScores)

#that fits VERY well:
np.mean(cvScores['test_score'])
print("Mean test score with only min_sentenceDistnace feature: ", np.mean(cvScores['test_score']))
# with features aggregated by context type:
#Out[7]: 0.96026284880758261
# With features NOT aggregated by context type:
#Mean test score with only min_sentenceDistnace feature:  0.849060858147

#-----------------------------------------------------------------------------------------------------#
# Train model using ALL features:
print("Training models and running CV with all features:")
X = df.loc[:, df.columns != 'label']
X = X.loc[:,X.columns != 'PMCID']
X = X.loc[:,X.columns != 'CtxID']
X = X.loc[:,X.columns != 'EvtID']
X = X.as_matrix()
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')
print("Cross-Val scores from all features: ", cvScores)
print("Mean CV test Score from all features: ", np.mean(cvScores['test_score']))
# With features NOT aggregated by context type:
#Mean CV test Score from all features:  0.827537216448

#cvScoresTestedByPaper = cross_validate(LR, X, y, groups = df['PMCID'], cv = 10, scoring = 'f1_micro')
#np.mean(cvScoresTestedByPaper['test_score'])


