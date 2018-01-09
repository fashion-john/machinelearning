#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 17:17
# @Author  : Aries
# @Site    : 
# @File    : 第九章.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
data=pd.read_csv('D:/machinelearninginaction/Ch09/ex00.txt',header=None,sep='\t')
# print(data.head())
X=data[0].reshape((-1,1))
y=data[1].values.reshape((-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_1=dtr.predict(X_test)
plt.figure()
plt.scatter(X,y,s=20,edgecolors="black",c="darkorange",label="data")
plt.plot(X_test,y_1,color="blue",linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title('DecisionTreereGressor')
plt.show()
