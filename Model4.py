#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 23:10:27 2018

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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

X_train_copy, X_valid_copy = X_train, X_valid
from sklearn.decomposition import PCA
pca = PCA(n_components= 9)
X_train_copy = pca.fit_transform(X_train_copy)
X_valid_copy = pca.fit_transform(X_valid_copy)
explained_variance = pca.explained_variance_ratio_

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

y_pred_valid = gbr.predict(X_valid)

from sklearn import metrics
metrics.mean_squared_error(y_valid, y_pred_valid)



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

X_test = sc.transform(X_test)

y_pred = gbr.predict(X_test)
df = pd.DataFrame()
df['ID'] = test_dataset['ID']
df['Upvotes'] = y_pred
df.to_csv('GBR_output.csv', index=False)