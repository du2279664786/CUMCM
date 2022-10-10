
import pandas as pd
import numpy as np
# from sklearn.linear_model import
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from math import log
import lightgbm as lgb
import os
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')
import winsound
import os
import gc
import re
import glob
# import pickle5 as pickle

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from urllib.parse import quote, unquote, urlparse

import lightgbm as lgb

data = pd.read_csv("E:/File/数学建模/2022数学建模/data/合并表一二筛选之后.csv")
data = data.drop(columns=['文物采样点','Unnamed: 0'])
feat = ['颜色','纹饰','类型','表面风化']
for i in feat:
    lb = LabelEncoder()
    data[i] = lb.fit_transform(data[i])

data = data.fillna(0)

train = data
feature_names = [i for i in data.columns.tolist() if i not in ['文物编号','类型']]
ycol = ['类型']

model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=2 ** 9,
                           max_depth=18,
                           learning_rate=0.1,
                           n_estimators=10000,
                           subsample=0.75,
                           feature_fraction=0.55,
                           reg_alpha=0.2,
                           reg_lambda=0.2,
                           random_state=i,     # 1983,852
                           is_unbalance=True,
                           # scale_pos_weight=130,
                           metric='auc')

df_importance_list = []

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1983)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    #         print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
print(df_importance)