
import warnings
warnings.filterwarnings('ignore')

# 导入包
from sklearn.cluster import KMeans   # 算法
from sklearn.datasets import load_iris     # 数据集
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.metrics import accuracy_score     #评估
from sklearn.preprocessing import StandardScaler   # 标准化
from sklearn.model_selection import GridSearchCV   # 交叉验证网格搜索(没用到）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

data = pd.read_csv("E:/File/数学建模/2022数学建模/data/归一化合并表一二筛选之后.csv",encoding='gb2312')

data1 = data[data['类型']=='高钾']
data2 = data[data['类型']=='铅钡']

feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data1[i] = lb.fit_transform(data1[i])

feats = [i for i in data1.columns.tolist() if i not in ['纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
# data.index = [i for i in range(0,49)]
data1.index = [i for i in range(0,18)]

feats = [i for i in data1.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
# data1 = data1[feats]
# data1[feats]

data_scaled = pd.DataFrame(data1[feats], columns=data1[feats].columns)


plt.figure(figsize=(10, 4))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/高钾的两个亚类划分化学元素')
plt.show()


# 铅钡层次分析图

feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data2[i] = lb.fit_transform(data2[i])

feats = [i for i in data2.columns.tolist() if i not in ['纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
data2.index = [i for i in range(0,49)]
# data2

feats = [i for i in data2.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
# data2 = data2[feats]
# data2[feats]

ta_scaled = pd.DataFrame(data2[feats], columns=data2[feats].columns)


plt.figure(figsize=(10, 4))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(ta_scaled, method='ward'))
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/铅钡的两个亚类划分化学元素')
plt.show()