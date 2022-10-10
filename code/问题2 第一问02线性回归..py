import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import json

import numpy as np
import pandas as pd
# pd.set_option('max_columns', None)
# pd.set_option('max_rows', 200)
# pd.set_option('float_format', lambda x: '%.3f' % x)
from tqdm.notebook import tqdm

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

data = pd.read_csv("E:/File/数学建模/2022数学建模/data/归一化合并表一二筛选之后.csv",encoding='gb2312')

feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])

train = data[['氧化铅(PbO)','二氧化硅(SiO2)','氧化钡(BaO)','氧化钾(K2O)']]
y = data['类型']

estimator = LogisticRegression()
estimator.fit(train,y)

print(estimator.coef_)    # w1-w4     氧化铅(PbO) 	二氧化硅(SiO2) 	氧化钡(BaO) 	氧化钾

print(estimator.intercept_)

test = data.iloc[[11,22,48,56]][['氧化铅(PbO)','二氧化硅(SiO2)','氧化钡(BaO)','氧化锶(SrO)']]
lab = data.iloc[[11,22,48,56]][['类型']]

estimator.predict(test)