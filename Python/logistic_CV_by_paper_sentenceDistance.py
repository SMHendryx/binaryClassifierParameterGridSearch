# File runs cross validation on binary logistic regression classifier using sklearn functions
#
# Author: Sean Hendryx
# 2017
# seanmhendryx@email.arizona.edu https://github.com/SMHendryx/binaryClassifierParameterGridSearch


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
        X_train = trainingSet['sentenceDistance'].reshape(-1, 1)
        y_train = trainingSet['label']
        #y_train = y_train.reshape((y_train.size, 1))
        X_test = testSet['sentenceDistance'].reshape(-1, 1)
        y_test = testSet['label']
        #y_test = y_test.reshape((y_test.size, 1))
        
        #del trainingSet['label']
        #X_train = trainingSet
        #remove PMCID:
        #del X_train['PMCID']
        # set up test data:
        #y_test = testSet['label']
        #del testSet['label']
        #X_test = testSet
        #del X_test['PMCID']

        # Normalize training data and get scaler: (not needed if we're only using 1 feature?)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        #train the model:
        # instantiate logistic regression object
        LR = LogisticRegression(penalty='l1', tol = LR_tolerance)
        LR.fit(X_train, y_train)
        
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
f1Scores_train, f1Scores_cv = cvByPaper(df)
#Scala LR_tolerance = 0.01

print("Macro Training Average:")
print(np.mean(f1Scores_train))
print("\n")
print("Macro CV Average:")
print(np.mean(f1Scores_cv))

