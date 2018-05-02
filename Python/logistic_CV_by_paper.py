# File runs cross validation on binary logistic regression classifier using sklearn functions
#
# Authored by Sean M. Hendryx while working at the University of Arizona
# seanmhendryx@email.arizona.edu 2017 https://github.com/SMHendryx/binaryClassifierParameterGridSearch


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
    f1_scores_train = np.zeros(len(paperIDs))
    f1_scores_cv = np.zeros(len(paperIDs))
    i = 0
    for paperID in paperIDs:
        print("Evaluating: ", paperID)

        #get Training Set:
        training_set = df.loc[df['PMCID'] != paperID]
        # test that the testSet has the correct number of rows:
        #df.loc[df['PMCID'] == paperID].shape[0] == df.shape[0] - training_set.shape[0]
        testSet = df.loc[df['PMCID'] == paperID]

        #get predictor variable in array (X_train) and response variable(y_train)
        y_train = training_set['label']
        del training_set['label']
        X_train = training_set
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
        LR = LogisticRegression(penalty='l2', C = 0.0001, tol = LR_tolerance).fit(X_train, y_train)
        
        # Compute predictions on train and test (cv):
        y_train_predicted = LR.predict(X_train)
        f1_score_train = metrics.f1_score(y_train, y_train_predicted)
        p = metrics.precision_score(y_train, y_train_predicted)
        r = metrics.recall_score(y_train, y_train_predicted)
        #Store the results:
        f1_scores_train[i] = f1_score_train

        y_test_predicted = LR.predict(X_test)
        f1_score_test = metrics.f1_score(y_test, y_test_predicted)
        f1_scores_cv[i] = f1_score_test
        
        print("Training F1")
        print(f1_score_train)

        print("CV F1")
        print(f1_score_test)
        i += 1

    return(f1_scores_train, f1_scores_cv)


# read in data:
in_file = '/Users/seanmhendryx/Data/context/features.feather'

df = feather.read_dataframe(in_file)

# Remove Context and Event IDs:
del df['EvtID']
del df['CtxID']


# run CV by paper:
# Run with these hyperperameters:
#In [5]: clf.best_params_
#Out[5]: {'C': 0.0001, 'penalty': 'l2'}
f1_scores_train, f1_scores_cv = cvByPaper(df)
# LR_tolerance = 0.01

print("Macro Training Average:")
print(np.mean(f1_scores_train))
print("\n")
print("Macro CV Average:")
print(np.mean(f1_scores_cv))
# Macro CV Average:
# 0.0689176818342

