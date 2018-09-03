#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:42:18 2018

@author: aniket
"""

##With the help of SVR
import numpy
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

#For training and validation data split
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.6, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

from xgboost import XGBRegressor
regressor = XGBRegressor

y_pred_valid = regressor.predict(X_valid)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_valid, y_pred_valid)