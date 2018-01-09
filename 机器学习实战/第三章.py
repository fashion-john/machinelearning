#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 15:15
# @Author  : Aries
# @Site    : 
# @File    : 第三章.py
# @Software: PyCharm
'''
使用决策树预测隐形眼镜类型
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
input_data=pd.read_csv('D:/machinelearninginaction/Ch03/lenses.txt',sep='\t',header=None)
dummiesX=input_data.drop([4],axis=1)
# 编码
lb=LabelEncoder()
for col in dummiesX.columns:
    dummiesX[col]=lb.fit_transform(dummiesX[col])

LabelY=input_data[4].values
# scikit-learn中决策树核心代码,采用ID3算法
clf=tree.DecisionTreeClassifier(criterion="entropy")
clf=clf.fit(dummiesX,LabelY)
# 可视化
dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,feature_names=dummiesX.keys(),class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('D:/machinelearninginaction/Ch03/result1.pdf')
