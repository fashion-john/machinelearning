#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 10:37
# @Author  : Aries
# @Site    : 
# @File    : 第八章.py
# @Software: PyCharm
'''
预测数值型数据：回归
回归的目的是预测数值型的目标值。
此处采用最小二乘法
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
data=pd.read_csv('D:/machinelearninginaction/Ch08/abalone.txt',sep='\t',header=None)
Y=data[8].values
X=data.drop([8],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
print(zip(X.columns,linreg.coef_))
print('accuracy is :',linreg.score(X_test,y_test))
rss=np.mean((linreg.predict(X_test)-y_test)**2)
print('RSS accuracy is :',rss)