#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 14:32
# @Author  : Aries
# @Site    : 
# @File    : 第五章.py
# @Software: PyCharm
'''
Logistic 回归
主要思想：根据现有数据对分类边界线建立回归公式，以此进行分类。
logistic regression 实际上不是一个回归器而是一个二分器，即给定的训练样本中一部分被标记为1，一部分被标记为0.我们从这些样本中训练出来一个分类器，给定输入特征。预测结果
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
test=pd.read_csv('D:/machinelearninginaction/Ch05/horseColicTest.txt',sep='\t',header=None)
train=pd.read_csv('D:/machinelearninginaction/Ch05/horseColicTraining.txt',sep='\t',header=None)
test_labels=test[21].values.tolist()
test_features=test.drop([21],axis=1)
train_labels=train[21].values
train_labels=train_labels.astype(int).tolist()
train_features=train.drop([21],axis=1)
# 数据预处理
# 缩放至0-1
min_max_scaler=MinMaxScaler()
test_features_after=min_max_scaler.fit_transform(test_features)
train_features_after=min_max_scaler.fit_transform(train_features)
X_train, X_test, y_train, y_test=train_test_split(train_features_after,train_labels,test_size=0.3,random_state=10)
# logistic model 模型
lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)
predict_result=lr_model.predict(X_test)
print('lr_accuracy:',lr_model.score(X_train,y_train))
# 随机森林模型
Rt_model=RandomForestClassifier()
Rt_model.fit(X_train,y_train)
predict_result=Rt_model.predict(X_test)
print('Rt_accuracy:',Rt_model.score(X_train,y_train))
'''
通过运行结果可知，以随机森林模型为代表的集成模型的效果要远远好于单模型，随机森林的分类准确率为0.97而logistic regression的分类准确率为0.71
'''