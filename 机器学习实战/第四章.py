#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 16:20
# @Author  : Aries
# @Site    : 
# @File    : 第四章.py
# @Software: PyCharm
'''
基于概率论的分类方法：朴素贝叶斯
基本思想：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。
'''
import pandas as pd
import os
from collections import Counter
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
path = 'C:/Users/john/Desktop/train-mails'
test_path = 'C:/Users/john/Desktop/test-mails'

# 获取频率最高的前3000个词汇
def make_Dictionary(path):
    emails = [path + '/' + j for j in os.listdir(path)]
    all_words = []
    for email in emails:
        with open(email) as f:
            for i, line in enumerate(f):
                if i == 2:
                    all_words += line.split()
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        # isalpha() 方法检测字符串是否只由字母组成
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

dictionary=make_Dictionary(path)

# 产生一个特征向量矩阵
def extract_features(path):
    files = [path + '/' + j for j in os.listdir(path)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0
    for file in files:
        with open(file) as f:
            for i, line in enumerate(f):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            docID += 1
    return features_matrix


train_label = np.zeros(702)
train_label[351:701] = 1
train_matrix = extract_features(path)
test_matrix=extract_features(test_path)
model2 = MultinomialNB()
model2.fit(train_matrix, train_label)
# test the unseen email
test_labels=np.zeros(260)
test_labels[130:259]=1
result2=model2.predict(test_matrix)
print(confusion_matrix(test_labels,result2))