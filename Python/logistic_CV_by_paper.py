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


def cvByPaper(df):
    """
    :param df: a pandas dataframe with a column named 'label', which is the response variable; 
        a column named PMCID, which is the paper id; 
        and the rest of the columns are features.
    :return: training and test scores
    """ 
    #make list of PMCIDs:
    paperIDs = pandas.unique(df.PMCID)

    #loop through folds (where each paper is a fold):
    f1Scores_train = np.zeros(len(paperIDs))
    f1Scores_cv = np.zeros(len(paperIDs))
    for(paperID in paperIDs):
        print("Evaluating: ", paperID)

        #get Training Set:
        trainingSet = df.loc[df['PMCID'] != paperID]
        # test that the testSet has the correct number of rows:
        #df.loc[df['PMCID'] == paperID].shape[0] == df.shape[0] - trainingSet.shape[0]
        testSet = df.loc[df['PMCID'] == paperID]

        #get predictor variable in array (X_train) and response variable(y_train)
        y_train = trainingSet['label']
        del trainingSet['label']
        X_train = trainingSet
        #remove PMCID:
        del X_train['PMCID']
        # set up test data:
        y_test = testSet['label']
        del testSet['label']
        X_test = testSet
        del X_test['PMCID']

        #train the model:
        # instantiate logistic regression object
        LR = LogisticRegression(penalty='l1').fit(X_train, y_train)
        # i am here:
        score = LR.score(X_test, y_test, scoring = 'f1_micro')





# read in data:
dir = '/Users/seanmhendryx/reach_context-balancing/reach'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)


# run CV by paper:




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
np.mean(scores)

#as opposed to cross_val_score, cross_validate returns a dict of float arrays of shape=(n_splits,), including 'test_score', 'train_score', 'fit_time', and 'score_time'
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')

#that fits VERY well:
np.mean(cvScores['test_score'])
#Out[7]: 0.96026284880758261

#let's now train model using all features:
X = df.ix[:, df.columns != 'label']
X = X.ix[:,X.columns != 'PMCID']
X = X.as_matrix()
cvScores = cross_validate(LR, X, y, cv = 10, scoring = 'f1_micro')

cvScoresTestedByPaper = cross_validate(LR, X, y, groups = df['PMCID'], cv = 10, scoring = 'f1_micro')
np.mean(cvScoresTestedByPaper['test_score'])


