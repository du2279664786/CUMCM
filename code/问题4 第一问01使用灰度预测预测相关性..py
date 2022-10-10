
import warnings
warnings.filterwarnings('ignore')

# 导入包
import gc
import seaborn as sns
from sklearn.cluster import KMeans   # 算法
from sklearn.datasets import load_iris     # 数据集
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.metrics import accuracy_score     #评估
from sklearn.preprocessing import StandardScaler   # 标准化
from sklearn.model_selection import GridSearchCV   # 交叉验证网格搜索(没用到）
import numpy as np
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import xgboost as xgb
import lightgbm as lgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

data = pd.read_csv("E:/File/数学建模/2022数学建模/data/归一化合并表一二筛选之后.csv",encoding='gb2312')
feat = ['颜色','纹饰','表面风化']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])
data_jia = data[data['类型']=='高钾']
data_bei = data[data['类型']=='铅钡']
feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]

result =pd.DataFrame(columns=[i for i in range(0,14)])

for tem in range(0,14):
    x=data_jia[feats].iloc[:,:].T
    x_mean=x.mean(axis=1)
    for i in range(x.index.size):
        x.iloc[i,:] = x.iloc[i,:]/x_mean[i]

    ck=x.iloc[tem,:]     # 二氧化挂
    a = [j for j in range(0,14)]
    cp=x.iloc[a,:]     # 剩下所有

    t=pd.DataFrame()
    for j in range(cp.index.size):
        temp=pd.Series(cp.iloc[j,:]-ck)
        t=t.append(temp,ignore_index=True)

    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho=0.4

    ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))

    ksi.columns.size
    r=ksi.sum(axis=1)/ksi.columns.size
    result[tem] = r
#     print(tem)
result.columns = feats
# result['类型'] = feats
result_jia = result

result.index = data_jia[feats].corr().index
mask = np.zeros_like(result, dtype=np.bool)   #定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(result)]= True      #返回矩阵的上三角，并将其设置为true
plt.figure(figsize=(20,16))
# cmap = sns.choose_diverging_palette()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(result,square=True, cmap=cmap,mask=mask,annot=True)
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/铅钡的灰度预测相关性')
plt.show()


result =pd.DataFrame(columns=[i for i in range(0,14)])

for tem in range(0,14):
    x=data_bei[feats].iloc[:,:].T
    x_mean=x.mean(axis=1)
    for i in range(x.index.size):
        x.iloc[i,:] = x.iloc[i,:]/x_mean[i]

    ck=x.iloc[tem,:]     # 二氧化挂
    a = [j for j in range(0,14)]
    cp=x.iloc[a,:]     # 剩下所有

    t=pd.DataFrame()
    for j in range(cp.index.size):
        temp=pd.Series(cp.iloc[j,:]-ck)
        t=t.append(temp,ignore_index=True)

    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho=0.4

    ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))

    ksi.columns.size
    r=ksi.sum(axis=1)/ksi.columns.size
    result[tem] = r
#     print(tem)
result.columns = feats
# result['类型'] = feats
result_bei = result

result.index = data_jia[feats].corr().index
mask = np.zeros_like(result, dtype=np.bool)   #定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(result)]= True      #返回矩阵的上三角，并将其设置为true
plt.figure(figsize=(20,16))
# cmap = sns.choose_diverging_palette()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(result,square=True, cmap=cmap,mask=mask,annot=True)
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/铅钡的灰度预测相关性')
plt.show()

result = result_jia-result_bei

result.index = data_jia[feats].corr().index
mask = np.zeros_like(result, dtype=np.bool)   #定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(result)]= True      #返回矩阵的上三角，并将其设置为true
plt.figure(figsize=(20,16))
# cmap = sns.choose_diverging_palette()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(result,square=True, cmap=cmap,mask=mask,annot=True)
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/高钾减去铅钡的相关性')
plt.show()