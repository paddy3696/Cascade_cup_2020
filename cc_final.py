#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 07:41:57 2020
@author: padmanabhan
"""
#%%
# import exploration files 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn 
import xgboost as xgb
#%%
file_path = 'train_age_dataset.csv'
# read in data 
data = pd.read_csv(file_path)
data_test = pd.read_csv('test_age_dataset.csv')
#%%
#Data Exploration

#rows and columns returns (rows, columns)
# data.shape
# data_test.shape
# #returns the first x number of rows when head(num). Without a number it returns 5
# data.head()
# #basic information on all columns 
# data.info()
# data_test.info()
#%%
y=data.iloc[:,[-1]]
x=data.drop(y.columns,axis = 1)
#%% 
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
x = x.iloc[:,0:]
#x = scaler.fit_transform(x)

data_test = data_test.iloc[:,0:]
#x_test = scaler.fit_transform(data_test) 
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
x_train,x_valid,y_train,y_valid = train_test_split(x,y, test_size=0.1)
sc = StandardScaler()
# sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
#x_train = x.to_numpy()
#y_train = y.to_numpy()
x_test = data_test.to_numpy()
x_test = sc.transform(x_test)
# x_train = x 
# y_train = y
print('dataset scaled')

#%% KNN
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train.values.ravel())
# print('model trained')
# y_pred1 = knn.predict(x_valid)

# knnf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred1, average = 'weighted')
# print('F1 score for KNN ', knnf1score)
# y_test_pred1 = knn.predict(x_test)
# np.savetxt("knn.csv", y_test_pred1)
#%%
# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTreeClassifier(max_depth=40, max_features=None, min_samples_leaf=40)

# dtree.fit(x_train,y_train)
# # y_pred2= dtree.predict(x_valid)
# # dtf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred2, average = 'weighted')
# # print('F1 score for Decision Tree ', dtf1score)
# y_test_pred2 = dtree.predict(x_test)
# np.savetxt("dtree.csv", y_test_pred2, header='prediction', comments='')
#%%
# from sklearn.svm import LinearSVC
# svm = LinearSVC(max_iter = 1000)
# svm.fit(x_train, y_train.values.ravel())
# y_pred3= svm.predict(x_valid)

# svmf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred3, average = 'weighted')
# print('F1 score for Linear SVC', svmf1score)
# y_test_pred3 = svm.predict(x_test)
# np.savetxt("svm.csv", y_test_pred3)
#%%
# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(max_depth = 40, min_samples_leaf=30, n_estimators=150)
# forest.fit(x_train, y_train.values.ravel())
# # y_pred4 = forest.predict(x_valid)
# # forestf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred4, average = 'weighted')
# # print('F1 score for Random Forest', forestf1score)
# y_test_pred4 = forest.predict(x_test)
# np.savetxt("forest.csv", y_test_pred4)
#%%
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators = 100, learning_rate = 1)
# ada.fit(x_train, y_train)
# y_pred5 = ada.predict(x_valid)
# adaf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred5, average = 'weighted')
# print('F1 score for Adaboost', adaf1score)
# y_test_pred5 = ada.predict(x_test)
# np.savetxt("ada.csv", y_test_pred5)
#%%
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# nb.fit(x_train, y_train)
# y_pred6 = nb.predict(x_valid)
# nbf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred6, average = 'weighted')
# print('F1 score for Naive Bayes', nbf1score)
# y_test_pred6 = nb.predict(x_test)
# np.savetxt("nb.csv", y_test_pred6)
#%% Logistic Reg
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(solver='newton-cg', multi_class='ovr',max_iter=1000)
# lr.fit(x_train,y_train.values.ravel())
# y_pred7 = lr.predict(x_valid)
# lrf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred7, average = 'weighted')
# print('F1 score for Logistic Regression', lrf1score)
# y_test_pred7 = lr.predict(x_test)
# np.savetxt("lr.csv", y_test_pred7)
#%%
from sklearn.ensemble import GradientBoostingClassifier
xg = xgb.XGBClassifier(n_estimators= 300, max_depth = 6, learning_rate = 0.5, min_child_weight=3,objective="multi:softmax")
#xg = GradientBoostingClassifier(n_estimators= 200, max_depth = 5, learning_rate = 0.5)
xg.fit(x_train,y_train.values.ravel())
y_pred8 = xg.predict(x_valid)
xgf1score = sklearn.metrics.f1_score(y_true = y_valid, y_pred = y_pred8, average = 'weighted')
print('F1 score for XG Boosting', xgf1score)
y_test_pred8 = xg.predict(x_test)
np.savetxt("xg.csv", y_test_pred8, header='prediction',fmt='%i', comments='')
