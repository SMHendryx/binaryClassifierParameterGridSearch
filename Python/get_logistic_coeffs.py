# Gets logistic coefficients for logistic regression trained on ALL context event relations
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


# read in data:
dir = '/Users/seanmhendryx/reach_context-balancing/reach'
inFile = 'features.feather'
# set wd:
os.chdir(dir)

df = feather.read_dataframe(inFile)
print("Data successfully read in.")

print("Training LR model with all features:")
# Remove Context and Event IDs:
del df['EvtID']
del df['CtxID']
X = df.ix[:, df.columns != 'label']
X = X.ix[:,X.columns != 'PMCID']
X = X.as_matrix()

# Normalize data:
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

y = df['label']
y = y.reshape((y.size, 1))

# instantiate logistic regression object
LR_tolerance = 0.01
LR = LogisticRegression(penalty='l1', tol = LR_tolerance)

#fit 
model = LR.fit(X, y)

model.coef_

dir = '/Users/seanmhendryx/githublocal/binaryClassifierParameterGridSearch/output'
# set wd:
os.chdir(dir)
np.savetxt('LR_model_coefficients_from_normalized_data.txt',model.coef_)

#Get feature names of the coefficients:
#https://stackoverflow.com/questions/34649969/how-to-find-the-features-names-of-the-coefficients-using-scikit-linear-regressio
del df['label']
del df['PMCID']
modelFeatures = df.columns
print(list(zip(model.coef_, modelFeatures)))

#get n greatest coefficients and their features:
n = 10
beta = model.coef_
ind = np.argpartition(beta, -n)[-n:]
beta[ind]
modelFeatures[ind]
modelFeatures[ind[np.argsort(beta[ind])]]




