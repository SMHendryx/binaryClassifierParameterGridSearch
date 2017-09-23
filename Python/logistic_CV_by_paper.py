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
from sklearn import preprocessing

def cvByPaper(df, LR_tolerance = 0.0001):
    """
    :param df: a pandas dataframe with a column named 'label', which is the response variable; 
        a column named PMCID, which is the paper id; 
        and the rest of the columns are features.
    :param LR_tolerance: sets the parameter of sklearn's LogisticRegression function (sklearn default is .0001), .01 leads to faster convergence.
    :return: training and test scores in two seperate numpy arrays
    """ 
    #make list of PMCIDs:
    paperIDs = pandas.unique(df.PMCID)

    #loop through folds (where each paper is a fold):
    f1Scores_train = np.zeros(len(paperIDs))
    f1Scores_cv = np.zeros(len(paperIDs))
    i = 0
    for paperID in paperIDs:
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

        # Normalize training data and get scaler:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        #train the model:
        # instantiate logistic regression object
        LR = LogisticRegression(penalty='l1', tol = LR_tolerance).fit(X_train, y_train)
        
        # Compute predictions on train and test (cv):
        y_train_predicted = LR.predict(X_train)
        f1Score_train = metrics.f1_score(y_train, y_train_predicted)
        p = metrics.precision_score(y_train, y_train_predicted)
        r = metrics.recall_score(y_train, y_train_predicted)
        #Store the results:
        f1Scores_train[i] = f1Score_train

        y_test_predicted = LR.predict(X_test)
        f1Score_test = metrics.f1_score(y_test, y_test_predicted)
        f1Scores_cv[i] = f1Score_test
        
        print("Training F1")
        print(f1Score_train)

        print("CV F1")
        print(f1Score_test)
        i += 1

    return(f1Scores_train, f1Scores_cv)


# read in data:
dir = '/Users/seanmhendryx/Data/context'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)

# Remove Context and Event IDs:
del df['EvtID']
del df['CtxID']


# run CV by paper:
# Run with these hyperperameters:
#In [5]: clf.best_params_
#Out[5]: {'C': 0.0001, 'penalty': 'l2'}
f1Scores_train, f1Scores_cv = cvByPaper(df, LR_tolerance = 0.01)

print("Macro Training Average:")
print(np.mean(f1Scores_train))
print("\n")
print("Macro CV Average:")
print(np.mean(f1Scores_cv))

# compute micro average:
paperIDs = pandas.unique(df.PMCID)
N = len(df)
i = 0
macroAverage_train = 0.0
macroAverage_test = 0.0
for paperID in paperIDs:
    n_i_test = len(df.loc[df['PMCID'] == paperID])
    #compute weighted averge chunk:
    av_w_i_test = (n_i_test/N) * f1Scores_cv[i]

    n_i_train = len(df.loc[df['PMCID'] != paperID])
    #compute weighted averge chunk:
    av_w_i_train = (n_i_train/N) * f1Scores_train[i]
    print("weighted average: ", av_w_i_train)

    macroAverage_train += av_w_i_train
    macroAverage_test += av_w_i_test
    i += 1






