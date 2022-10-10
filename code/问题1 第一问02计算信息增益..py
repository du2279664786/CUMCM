
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from math import log
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

file1 = pd.read_excel('E:/File/数学建模/2022数学建模/data/附件.xlsx',sheet_name = 0)
file2 = pd.read_excel('E:/File/数学建模/2022数学建模/data/附件.xlsx',sheet_name = 1)

file1 = file1.fillna('NAN')

def chane_col(x):
    dic = {'绿':0, '黑':1, '紫':2, '深绿':3, '深蓝':4, 'NAN':5, '浅绿':6, '浅蓝':7, '蓝绿':8}
    return dic[x]

def chane_wenshi(x):
    dic = {'A':0, 'B':1, 'C':2}
    return dic[x]

def chane_leixing(x):
    dic = {'高钾':0, '铅钡':1}
    return dic[x]

def chane_fenghua(x):
    dic = {'无风化':0, '风化':1}
    return dic[x]

file1['颜色'] = file1['颜色'].apply(lambda x:chane_col(x))
feat = [i for i in file1.columns.tolist() if i not in ['颜色']]
for i in feat:
    lb = LabelEncoder()
    file1[i] = lb.fit_transform(file1[i])

file1 = file1[['纹饰', '类型','颜色','表面风化']]

# 信息熵
def info_entropy(attr):
    prob = pd.value_counts(attr) / len(attr)   # 对于一个特征不同类所占的比例类
    return sum( np.log2( prob )* prob * (-1) )  # 经验熵

for i in ['纹饰', '类型','颜色','表面风化']:
    print(str(i)+'特征的信息熵:'+str(round(info_entropy(file1[i]),4)))

# 信息增益   （返回值越大，attr1 与 attr2 相关性越强）
def info_gain(dataset, attr1, attr2):
    ent1= dataset.groupby(attr1).apply(lambda x: info_entropy(x[attr2]))
    prob = pd.value_counts(dataset[attr1]) / len(dataset[attr1])
    ent2= sum( ent1 * prob )                   # 经验条件熵
    return info_entropy(dataset[attr2]) - ent2     #  信息增益
for i in ['纹饰', '类型','颜色','表面风化']:
#     print(i,info_gain(file1,'表面风化', i))
    print(str(i)+'特征的信息增益:'+str(round(info_gain(file1,'表面风化', i),4)))

# 3.
def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))
# 定义计算信息增益的函数：计算g(D|A)
def g(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: infor(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    # 计算Infor(D|A)
    e2 = sum(e1 * p1)
    return infor(data[str2]) - e2
# print("学历信息增益：{}".format(g(file1, "表面风化", "颜色")))
for j in ['表面风化','纹饰', '类型','颜色','表面风化']:
    for i in ['表面风化','纹饰', '类型','颜色','表面风化']:
#     print(i,info_gain(file1,'表面风化', i))
        print(str(j)+'和'+str(i)+'特征的信息增益:'+str(round(g(file1,j, i),4)/np.sqrt(round(info_entropy(file1[i]),4)*round(info_entropy(file1[j]),4))))