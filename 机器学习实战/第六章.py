#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 16:00
# @Author  : Aries
# @Site    : 
# @File    : 第六章.py
# @Software: PyCharm
'''
SVM算法：手写数字识别
'''
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

train_path = 'D:/machinelearninginaction/Ch02/trainingDigits'
test_path = 'D:/machinelearninginaction/Ch02/testDigits'
train_location = [train_path + '/' + j for j in os.listdir('D:/machinelearninginaction/Ch02/trainingDigits')]
test_location = [test_path + '/' + j for j in os.listdir('D:/machinelearninginaction/Ch02/testDigits')]
train_label = [int(j[0]) for j in os.listdir('D:/machinelearninginaction/Ch02/trainingDigits')]
test_label = [int(j[0]) for j in os.listdir('D:/machinelearninginaction/Ch02/testDigits')]


# 观察文本可知，文本是32X32格式
def img2vector(filename):
    retrunMetrix = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            retrunMetrix[0, 32 * i + j] = int(line[j])
    return retrunMetrix


train_features = np.zeros((len(train_location), 1024))
test_features = np.zeros((len(test_location), 1024))
for i in range(len(train_location)):
    train_features[i, :] = img2vector(train_location[i])
for i in range(len(test_location)):
    test_features[i, :] = img2vector(test_location[i])
X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.3, random_state=42)
'''
在sklearn中svm有三个类,分别是SVC,NuSVC,LinearSVC
SVC和NuSVC类似，但是接受的参数不同，而且底层的数学原理不一样。
LinearSVC是对支持向量分类的另一种实现，使用了线性核，但是不接受关键字kernel
前两类使用‘一对多’方法来实现多类别分类
支持向量回归有三种不同实现：SVR, NuSVR和LinearSVR。
LinearSVR提供的实现比SVR快，但是只使用线性核。
NuSVR则是使用了一个略不同的数学原理。
'''

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
print('accuracy:', svm_model.score(X_test, y_test))
