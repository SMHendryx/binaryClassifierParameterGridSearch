# File tests logistic regression parameter settings
# Authored by Sean M. Hendryx while working at the University of Arizona
# contact: seanmhendryx@email.arizona.edu
# Add licencse and copyright


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
dir = '/Users/seanmhendryx/reach_context-balancing/reach'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)

# set up data:
X = df['min_sentenceDistance']
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
scores = cross_val_score(LR, X, y, cv=10, scoring='f1_micro')

#as opposed to cross_val_score, cross_validate returns a dict of float arrays of shape=(n_splits,), including 'test_score', 'train_score', 'fit_time', and 'score_time'
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')

#that fits VERY well:
np.mean(cvScores['test_score'])
#Out[7]: 0.96026284880758261

#let's now try to replicate poor results by using all other features:
X = df.ix[:, df.columns != 'label']
X = X.ix[:,X.columns != 'PMCID']
X = X.as_matrix()
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')

cvScoresTestedByPaper = cross_validate(LR, X, y, groups = df['PMCID'], cv = 10, scoring = 'f1_micro')



