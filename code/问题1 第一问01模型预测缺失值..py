
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from math import log

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

file1 = pd.read_excel('E:/File/数学建模/2022数学建模/data/附件.xlsx',sheet_name = 0)
# file2 = pd.read_excel('E://File//数学建模//新建文件夹//C题/附件.xlsx',sheet_name = 1)
# file3 = pd.read_excel('E://File//数学建模//新建文件夹//C题/附件.xlsx',sheet_name = 2)
file1 = file1.fillna('NAN')

def chane_col(x):
    dic = {'蓝绿':0, '浅蓝':1, '紫':2, '深绿':3, '深蓝':4, 'NAN':5, '浅绿':6, '黑':7, '绿':8}
    return dic[x]
file1['颜色'] = file1['颜色'].apply(lambda x:chane_col(x))

feat = [i for i in file1.columns.tolist() if i not in ['颜色','文物编号']]
for i in feat:
    lb = LabelEncoder()
    file1[i] = lb.fit_transform(file1[i])

file1 = file1[['纹饰', '类型','颜色','表面风化']]
train = file1[file1['颜色']!=5]
test = file1[file1['颜色']==5]

# 训练模型
estimator = svm.SVC()
estimator.fit(train[feat],train['颜色'])
# 预测模型
print(estimator.predict(test[feat]))