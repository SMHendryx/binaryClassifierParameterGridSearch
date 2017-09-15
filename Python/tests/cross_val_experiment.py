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
from sklearn.model_selection import StratifiedKFold


# read in data:
dir = '/Users/seanmhendryx/reach_context-balancing/reach'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)
print("Data successfully read in.")

print("Training models and running CV with only min_sentenceDistance feature:")
# set up data:
X = df['min_sentenceDistance']
y = df['label']
X = X.reshape((X.size, 1))
y = y.reshape((y.size, 1))

shouldBeMaxIndexInFirstFold = X.size/10

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)

print(skf)
# |  StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    break





from sklearn.model_selection import StratifiedKFold
X1 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y1 = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)  # doctest: +NORMALIZE_WHITESPACE
# |  StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
