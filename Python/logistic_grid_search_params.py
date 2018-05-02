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
from sklearn.model_selection import GridSearchCV


# read in data:
dir = '/Users/seanmhendryx/Data/context'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)
print("Data successfully read in.")

print("Training models and running CV with only sentenceDistance feature:")

# SET UP DATA:
#sentenceDistance is the distance between the event and context mention.
X = df['sentenceDistance']
y = df['label']
X = X.reshape((X.size, 1))
y = y.reshape((y.size, 1))

#Apparently cross_val_score  requires an (R,) shape label instead of (R,1) (which is confusing bc LogisticRegression requires the opposite):
c, r = y.shape
del r
y = y.reshape(c,)
#scores = cross_val_score(LR, X, y, cv=10, scoring='f1_micro')
#np.mean(scores)

#VALIDATE
# instantiate logistic regression object
LR = LogisticRegression(penalty='l1')

#fit 
#model = LR.fit(X, y)

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
not_features = ['PMCID', 'label', 'EvtID', 'CtxID']
X = df.loc[:, df.columns not in not_features]
X = X.loc[:,X.columns != 'PMCID']
X = X.loc[:,X.columns != 'CtxID']
X = X.loc[:,X.columns != 'EvtID']
X = X.as_matrix()

#Set up hyperparameters to grid search through:
Cs = np.logspace(-4, 4, 3)
parameters = {'penalty':('l2', 'l1'), 'C':Cs}

clf = GridSearchCV(LR, parameters)
clf.fit(X, y)

#view results
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


cvScores = cross_validate(clf, X, y, cv = 10, scoring = 'f1_micro')
print("Cross-Val scores from all features: ", cvScores)
print("Mean CV test Score from all features: ", np.mean(cvScores['test_score']))
# With features NOT aggregated by context type:
#Mean CV test Score from all features:  0.827537216448


#------------------------------------------------------------------------------------#
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

Cs = np.logspace(-4, 4, 3)
parameters = {'penalty':('l2', 'l1'), 'C':Cs}

clf = GridSearchCV(LR, parameters)
clf.fit(X_train, y_train)

#view results
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

sorted(clf.cv_results_.keys())







