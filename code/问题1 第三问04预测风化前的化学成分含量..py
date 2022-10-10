

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from math import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("E:/File/数学建模/2022数学建模/归一化合并表一二筛选之后.csv",encoding='gb2312')
feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])
data = data.fillna(0)
data_jia = data[data['类型']==1]
data_bei = data[data['类型']==0]

feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
data_jia_hou_qian = pd.DataFrame()
data_jia_wei = data_jia[data_jia['表面风化']==0]
data_jia_hou = data_jia[data_jia['表面风化']==1]
dic = {}
for i in data_jia[feats]:
    # 风化前的平均
#     print(i,data_jia.groupby('表面风化')[i].mean().tolist())
#     qian  = data_jia.groupby('表面风化')[i].mean().tolist()[0]

#     # 风化后的平均
#     hou  = data_jia.groupby('表面风化')[i].mean().tolist()[1]
#     # 高钾风化前和风化后的差值
    chazhi = data_jia.groupby('表面风化')[i].mean().tolist()[0]-data_jia.groupby('表面风化')[i].mean().tolist()[1]
    # 计算标准化
    scal = (data_jia_hou[i]-data_jia_hou[i].mean())/data_jia_hou[i].std()
#     scal = (data_jia_hou[i]-data_jia_hou[i].mean())/data_jia_hou[i].std()
    # 归一化
    one = (data_jia_hou[i]-data_jia_hou[i].min())/(data_jia_hou[i].max()-data_jia_hou[i].min())
#     one = data_jia_hou[i]/data_jia_hou[i].mean()
#     res = scal*chazhi
#     print(chazhi)
#     print(i,(data_jia[i]-data_jia[i].mean()).mean())
#     print(i,round(res*100,5))
    data_jia_hou_qian[i] = data_jia_hou[i]+scal*one/10+chazhi
#     print(scal*one/10)
data_jia_hou_qian = data_jia_hou_qian.fillna(0)
# data_jia_hou_qian.to_csv('E:/File/数学建模/新建文件夹/问题1-3高钾风化前化学成分含量.csv',index=False)
data_jia_hou_qian.to_csv("E:/File/数学建模/2022数学建模/data/问题一预测的高钾.csv")
print(data_jia_hou_qian)

feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]
data_bei_hou_qian = pd.DataFrame()
data_bei_wei = data_bei[data_bei['表面风化']==0]
data_bei_hou = data_bei[data_bei['表面风化']==1]
dic = {}
for i in data_bei[feats]:
    # 风化前的平均
#     print(i,data_bei.groupby('表面风化')[i].mean().tolist())
#     qian  = data_bei.groupby('表面风化')[i].mean().tolist()[0]

#     # 风化后的平均
#     hou  = data_bei.groupby('表面风化')[i].mean().tolist()[1]
#     # 高钾风化前和风化后的差值
    chazhi = data_bei.groupby('表面风化')[i].mean().tolist()[0]-data_bei.groupby('表面风化')[i].mean().tolist()[1]
    # 计算标准化
    scal = (data_bei_hou[i]-data_bei_hou[i].mean())/data_bei_hou[i].std()
#     scal = (data_bei_hou[i]-data_bei_hou[i].mean())/data_bei_hou[i].std()
    # 归一化
    one = (data_bei_hou[i]-data_bei_hou[i].min())/(data_bei_hou[i].max()-data_bei_hou[i].min())
#     one = data_bei_hou[i]/data_bei_hou[i].mean()
#     res = scal*chazhi
#     print(chazhi)
#     print(i,(data_bei[i]-data_bei[i].mean()).mean())
    print(data_bei)
    data_bei_hou_qian[i] = data_bei_hou[i]+scal*one/10+chazhi
#     print(scal*one/10)
# data_bei_hou_qian = data_bei_hou_qian.fillna(0)
# # data_bei_hou_qian.to_csv('E:/File/数学建模/新建文件夹/问题1-3铅钡风化前化学成分含量.csv',index=False)
# data_jia_hou_qian.to_csv("E:/File/数学建模/2022数学建模/data/问题一预测的铅钡.csv")
print(data_bei_hou_qian)