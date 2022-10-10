
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os


plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

data = pd.read_csv("E:/File/数学建模/2022数学建模/data/亚分类.CSV",encoding='gb2312')
data = data.drop(columns=['文物采样点','Unnamed: 0'])
data_jia=data[data['类型']=='高钾']
data_bei=data[data['类型']=='铅钡']

feat = ['颜色','纹饰','表面风化']
for i in feat:
    lb = LabelEncoder()
    data_jia[i] = lb.fit_transform(data_jia[i])
data_jia = data_jia.fillna(0)

def chan(x):
    if x==0:
        return '亚类1'
    else:
        return '亚类2'

data_jia['label'] = data_jia['label'].apply(lambda x:chan(x))

feats = [i for i in data_jia.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0','label']]
# data = data[data['类型']==1]
X = data_jia[feats]
y = data_jia['label']

# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)
# data[feats]

text_representation = tree.export_text(clf)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    clf,
    feature_names=feats,
    class_names=['亚类1','亚类2'],
    filled=True
)
plt.show()
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/决策树实现高钾的两个亚类划分化学元素')

feat = ['颜色','纹饰','表面风化']
for i in feat:
    lb = LabelEncoder()
    data_bei[i] = lb.fit_transform(data_bei[i])
data_bei = data_bei.fillna(0)


def chan(x):
    if x==3:
        return '亚类1'
    else:
        return '亚类2'

data_bei['label'] = data_bei['label'].apply(lambda x:chan(x))



feats = [i for i in data_bei.columns.tolist() if i not in ['文物编号','纹饰','类型','颜色','表面风化','文物采样点','Unnamed: 0','label']]
# data = data[data['类型']==1]
X = data_bei[feats]
y = data_bei['label']

# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)
# data[feats]


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    clf,
    feature_names=feats,
    class_names=['亚类3','亚类4'],
    filled=True
)
plt.show()
plt.savefig('E:/File/数学建模/2022数学建模/图片文件/决策树实现铅钡的两个亚类划分化学元素')
