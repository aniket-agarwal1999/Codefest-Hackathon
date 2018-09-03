#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:09:54 2018

@author: aniket
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('train.csv')

X = train_dataset.iloc[:, [1,2,3,5]].values
y = train_dataset.iloc[:, 6].values

#Encoding the category of the questions column
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#So as to remove the dependant variable paradox
X = X[:, 1:]

#This after the scaling 
X = 

'''
#Here the PCA for reduction of features
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1), dtype='int'), values=X, axis = 1)

####This is some highly shitted code for the prob
#Now we are gonna check the P-values of the various features
X_opt = X[:,:]
regressor_OLS = sm.OLS(endog = y,  exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,3,4,5,6,7,9,10,11,12]]
regressor_OLS = sm.OLS(endog = y,  exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,3,4,5,7,9,10,11, 12]]
regressor_OLS = sm.OLS(endog = y,  exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y,  exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,3]]

'''


#For training and validation data split
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 0)

#For scaling Reputation, Answers, Views
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)