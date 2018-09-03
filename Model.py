#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:49:21 2018

@author: aniket
"""

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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 0)

#For scaling Reputation, Answers, Views
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_valid)

from sklearn import metrics
metrics.mean_squared_error(y_valid, y_pred)



#####From here the we start the testing phase
test_dataset = pd.read_csv('test.csv')
X_test = test_dataset.iloc[:, [1,2,3,5]].values

#Encoding the category of the questions column
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
X_test[:, 0] = labelencoder.fit_transform(X_test[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_test = onehotencoder.fit_transform(X_test).toarray()

#So as to remove the dependant variable paradox
X_test = X_test[:, 1:]

#For scaling Reputation, Answers, Views
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

y_pred_test = regressor.predict(X_test)


df= pd.DataFrame()
test_out = pd.read_csv('test.csv')
df['ID'] = test_out['ID']
df['Upvotes'] = y_pred_test
df.to_csv('output.csv', index=False)



