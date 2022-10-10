
import warnings
warnings.filterwarnings('ignore')

# 导入包
import gc
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
test = pd.read_excel('E://File//数学建模/2022数学建模/data/附件.xlsx',sheet_name = 2)

test = test.fillna(0)
lb = LabelEncoder()
test['表面风化'] = lb.fit_transform(test['表面风化'])

feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])

feats = [i for i in test.columns.tolist() if i not in ['文物编号']]

X = data[feats]
test = test[feats]
y = data['类型']

clf = RandomForestClassifier()
clf.fit(X, y)#训练

print(clf.predict(test))