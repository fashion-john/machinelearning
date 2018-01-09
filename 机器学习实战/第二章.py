#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 10:04
# @Author  : Aries
# @Site    : 
# @File    : 第二章.py
# @Software: PyCharm
'''
KNN算法
工作原理：存在一个样本数据集合，也称为训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最相邻）的分类标签。一般来说，我们只选择样本数据集中前K个最相似的数据，这就是k-近邻算法中的k的出处，通常k是不大于20的整数。最后，选择K个最相似数据中出现次数最多的分类，作为新数据的分类
'''
import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
inputText=pd.read_csv('D:/machinelearninginaction/Ch02/datingTestSet.txt',sep='\t',header=None)
targets=inputText[3].values.tolist()
features=inputText.drop([3],axis=1)
# 编码，此处只是简单的将分类标签变成数字表示，所以不必使用preprocessing函数
for i in range(len(targets)):
    if targets[i]=='largeDoses':
        targets[i]=1
    elif targets[i]=='smallDoses':
        targets[i]=2
    else:
        targets[i]=3
# 归一化处理 对应程序清单2-3
min_max_scaler=preprocessing.MinMaxScaler()
X_train_scaler=min_max_scaler.fit_transform(features)
# 划分训练集和测试集
X_train,X_test,Y_train,Y_test=train_test_split(features,targets,test_size=0.3)
# scikit-learn 中KNN算法核心程序
knn=neighbors.KNeighborsClassifier()
knn.fit(X_train,Y_train)
predict=knn.predict(X_test)
# print(predict)
print("accuracy is:",knn.score(X_train,Y_train))