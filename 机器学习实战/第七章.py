#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 14:39
# @Author  : Aries
# @Site    : 
# @File    : 第七章.py
# @Software: PyCharm
'''
Adaboost算法
思想：boosting分类的结果是基于所有分类器的加权求和结果的，因此boosting和bagging不大一样。
bagging中的分类器权重是相等的，而boosting的分类器权重是不相等的，每个权重代表的是对应分类器在上一轮迭代中的
成功度
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
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
# scikit-learn中adaboost算法核心程序
ada_model=AdaBoostClassifier(n_estimators=10)
ada_model.fit(X_train,y_train)
'''
因为predict_proba返回的是一个两列的矩阵，矩阵的每一行代表的是对一个事件的预测结果，第一列代表该事件不会发生的概率，第二列代表的是该事件会发生的概率。
而这里需要的是第二列的数据
'''
X_predict=ada_model.predict_proba(X_test)
print('accuracy:',ada_model.score(X_test,y_test))
# 绘制ROC曲线
f_pos,t_pos,thresh=roc_curve(y_test,X_predict[:,1],pos_label=1)
auc_area=auc(f_pos,t_pos)
plt.plot(f_pos,t_pos,'darkorange',lw=2,label='AUC=%0.2f'%auc_area)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.title('ROC-AUC')
plt.ylabel('True Pos Rate')
plt.xlabel('False Pos Rate')
plt.show()