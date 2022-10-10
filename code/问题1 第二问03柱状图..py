
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


plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

pd.set_option('display.max_columns', None)

file1 = pd.read_excel('E:/File/数学建模/2022数学建模/data/附件-填充.xlsx',sheet_name = 0)
file2 = pd.read_excel('E:/File/数学建模/2022数学建模/data/附件-填充.xlsx',sheet_name = 1)

file2['文物编号'] = file2['文物采样点'].apply(lambda x:int(str(x)[0:2]))
data = pd.merge(file1,file2,on='文物编号')

data = data[data[[i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点']]].sum(axis=1)>85]
data.to_csv("E:/File/数学建模/2022数学建模/data//合并表一二筛选之后.csv",encoding='utf8')

feat = ['颜色','纹饰','类型','表面风化','文物采样点']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])
data = data.fillna(0)
data = pd.read_csv("E:/File/数学建模/2022数学建模/合并表一二筛选之后.csv",encoding='gb2312')

data_jia = data[data['类型']=='高钾']
data_bei = data[data['类型']=='铅钡']

feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]

# plt.figure(figsize=(40,16))
# j = 0
# for i in data_jia[feats]:
#     j+=1
#     plt.subplot(3,5,j)
#     x = data_jia.groupby('表面风化')[i].mean().tolist()
#     y = ['无风化','风化']
# #     sns.barplot(y,x)
#     plt.bar(y,x)
#     plt.title(i)
# plt.show()

feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]

plt.figure(figsize=(50,30))
j = 0
for i in data_jia[feats]:
    j+=1
    plt.subplot(3,5,j)
    x = data_jia.groupby('表面风化')[i].mean().tolist()
    y = ['无风化','风化']
    sns.barplot(x=y,y=x)
#     plt.bar(y,x)
#     plt.title(i)
    # 设置刻度字体大小
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
#     # 设置坐标标签字体大小
    plt.xlabel(i,fontsize=45)
#     plt.set_ylabel(..., fontsize=20)
    # 设置图例字体大小
#     plt.legend(..., fontsize=20)


# plt.savefig('E:/File/数学建模/新建文件夹/图片文件/问题一/高钾的风化和非风化对比')
plt.show()



# feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]

# plt.figure(figsize=(40,16))
# j = 0
# for i in data_jia[feats]:
#     j+=1
#     plt.subplot(3,5,j)
#     x = data_jia.groupby('表面风化')[i].mean().tolist()
#     y = ['无风化','风化']
# #     sns.barplot(y,x)
#     plt.bar(y,x)
#     plt.title(i)
# plt.show()

feats = [i for i in data.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0']]

plt.figure(figsize=(50,30))
j = 0
for i in data_bei[feats]:
    j+=1
    plt.subplot(3,5,j)
    x = data_bei.groupby('表面风化')[i].mean().tolist()
    y = ['无风化','风化']
    sns.barplot(x=y,y=x)
#     plt.bar(y,x)
#     plt.title(i)
    # 设置刻度字体大小
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
#     # 设置坐标标签字体大小
    plt.xlabel(i,fontsize=45)
#     plt.set_ylabel(..., fontsize=20)
    # 设置图例字体大小
#     plt.legend(..., fontsize=20)


# plt.savefig('E:/File/数学建模/新建文件夹/图片文件/问题一/铅钡的风化和非风化对比')
plt.show()
