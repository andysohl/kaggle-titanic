import gc
gc.collect()

import time
import pandas as pd
import numpy as np
import seaborn as sns
#sns.set(rc={'figure.figsize':(20,10)})
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
lbec = LabelEncoder()
scaler = StandardScaler()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, plot_precision_recall_curve
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_tree, XGBClassifier, plot_importance
import lightgbm as lgb
from lightgbm import plot_importance as lgbm_plt
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
# import pycaret

import re
import pickle
import os
os.chdir(r"C:\Users\andyhlso\Desktop\Andy\Kaggle\202209 - spaceship titanic")

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df2 = df.copy()
df2 = df2.set_index('PassengerId')
df2_test = df_test.copy()
df2_test = df2_test.set_index('PassengerId')

df_all = pd.concat([df2,df2_test])

drop_vars = ['Name']
df2 = df2.drop(drop_vars, axis=1)
df2_test = df2_test.drop(drop_vars, axis=1)

df2['Transported'].value_counts(dropna=False)

df_all.columns
df_all.dtypes
"""======================================== Data cleaning ========================================"""
# def check_missing_row(missing_rate):   
#     missing_row = (df2.isnull().sum(axis=1))/len(df2.columns) *100
#     chk = df2.loc[missing_row[missing_row >= missing_rate].index]
#     print('Missing rows pct > ' + str(missing_rate) + ' by Group:')
#     print(chk.value_counts())

# def remove_missing_row_data(df, missing_rate):
#     #remove data with 60% missing
#     #most of missing rows are inactive customers, so remove them
#     #check missing value by rows
#     missing_row = (df.isnull().sum(axis=1))/len(df.columns) *100
#     return(df.loc[missing_row[missing_row < missing_rate].index])


def check_missing_col(df):
    #check missing values by column
    missing = df.isnull().sum()/ len(df) *100
    missing = missing[missing > 0]
    return(missing)

def remove_missing_col_data(df, missing_rate):
    missing = df_all.isnull().sum()/ len(df_all) *100
    missing = missing[missing > 0]
    #fill 0 withs missing percentage greater than 80
    missing80pct_cols = missing[missing > missing_rate].index.tolist()
    #df[missing80pct_cols] = df[missing80pct_cols].fillna(0)
    print('no of cols to drop: ' + str(len(missing80pct_cols)))
    return(df.drop(missing80pct_cols, axis=1))

check_missing_col(df_all)
# HomePlanet       3.313010 D
# CryoSleep        3.566088 D
# Cabin            3.439549 D
# Destination      3.151961 D
# Age              3.105947 D
# VIP              3.405039 D
# RoomService      3.025423 D
# FoodCourt        3.324514 D
# ShoppingMall     3.520074 D
# Spa              3.266996 D
# VRDeck           3.082940 D


df_all['HomePlanet'].value_counts()
# Earth     6865
# Europa    3133
# Mars      2684

df_all['Destination'].value_counts()
# TRAPPIST-1e      8871
# 55 Cancri e      2641
# PSO J318.5-22    1184

df_all['CryoSleep'].value_counts()
# False    8079
# True     4581

df2['HomePlanet'] = df2['HomePlanet'].fillna('Earth')
df2_test['HomePlanet'] = df2_test['HomePlanet'].fillna('Earth')
df2['Destination'] = df2['Destination'].fillna('TRAPPIST-1e')
df2_test['Destination'] = df2_test['Destination'].fillna('TRAPPIST-1e')
df2['CryoSleep'] = df2['CryoSleep'].fillna(False)
df2_test['CryoSleep'] = df2_test['CryoSleep'].fillna(False)


df2['Age'] = df2['Age'].fillna(df_all['Age'].mean())
df2_test['Age'] = df2_test['Age'].fillna(df_all['Age'].mean())

mapping = {True:1,False:0}
df2['VIP'] = df2['VIP'].map(mapping)
df2_test['VIP'] = df2_test['VIP'].map(mapping)
df2['VIP'] = df2['VIP'].fillna(0)
df2_test['VIP'] = df2_test['VIP'].fillna(0)


df2['CryoSleep'] = df2['CryoSleep'].map(mapping)
df2_test['CryoSleep'] = df2_test['CryoSleep'].map(mapping)

df2['Transported'] = df2['Transported'].map(mapping)


dollar_vars = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for var in dollar_vars:
    df2[var] = df2[var].fillna(0)
    df2_test[var] = df2_test[var].fillna(0)

df2[['cabin_deck','cabin_num','cabin_side']] = df2['Cabin'].str.split('/',expand=True)
df2_test[['cabin_deck','cabin_num','cabin_side']] = df2_test['Cabin'].str.split('/',expand=True)

df2['cabin_num'] = pd.to_numeric(df2['cabin_num'])
df2_test['cabin_num'] = pd.to_numeric(df2_test['cabin_num'])

df2 = df2.dropna(subset='Cabin')
# df2_test = df2_test.dropna(subset='Cabin')
# df_all[df_all['Cabin'].isna()]['Transported'].value_counts()
# True     100
# False     99

df2 = df2.drop('Cabin',axis=1)
df2_test = df2_test.drop('Cabin',axis=1)


df2['cabin_deck'].value_counts()
# F    2794
# G    2559
# E     876
# B     779
# C     747
# D     478
# A     256
# T       5

df2['cabin_side'].value_counts()
# S    4288
# P    4206

one_hot_vars = ['HomePlanet','cabin_deck','cabin_side','Destination']
for var in one_hot_vars:
    df2 = pd.get_dummies(df2, columns=[var], prefix = ['OH_' + var])
    df2_test = pd.get_dummies(df2_test, columns=[var], prefix = ['OH_' + var])


scale_vars = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Age','cabin_num']
df2[scale_vars] = scaler.fit_transform(df2[scale_vars])
df2_test[scale_vars] = scaler.fit_transform(df2_test[scale_vars])

# label_vars = ['CryoSleep','VIP']
# for var in label_vars:   
#     df2['label_'+var] = lbec.fit_transform(df2[var].astype(str))
#     df2_test['label_'+var] = lbec.fit_transform(df2_test[var].astype(str))
#     df2 = df2.drop([var],axis=1)
        
corr_df = df2.corr().abs()




# def remove_correlated(df):
#     corr_df = df.corr().abs()
#     # Create and apply mask
#     mask = np.triu(np.ones_like(corr_df, dtype=bool))
#     tri_df = corr_df.mask(mask)
#     tri_df.to_excel('df_correlation.xlsx')
#     # Find columns that meet treshold
#     to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.89)]
#     print(to_drop)
#     #df = df.drop(to_drop, axis=1)
#     print("removed correlation >0.89")
#     return(df.drop(to_drop, axis=1))
           
# df = remove_correlated(df)

#_______modelling_________#
X = df2.drop('Transported', axis=1)
y = df2['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("LGBM",lgb.LGBMClassifier()))

for name,model in models:
    kfold = KFold(n_splits=5, random_state=22, shuffle=True)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "f1")
    print(name, cv_result)
    
# LR [0.80443828 0.7991453  0.76889215 0.7958884  0.80755773]
# NB [0.72757112 0.78883495 0.7604712  0.75790833 0.77928483]
# RF [0.7901419  0.78300455 0.77065637 0.79657054 0.78959108]
# SVC [0.8258427  0.80996441 0.78164323 0.81029412 0.80763791]
# Dtree [0.75520459 0.75358166 0.74536006 0.74391144 0.75142857]
# XGB [0.79741379 0.79731744 0.77317257 0.79245283 0.79857651]
# KNN [0.7798098  0.76888889 0.77211394 0.78592814 0.76977904]
# LGBM [0.80545585 0.79056047 0.77918216 0.80363912 0.80399145]
    

# """lightGBM"""
clf = lgb.LGBMClassifier(objective=' y',
                          metric='logloss',
                          learning_rate = 0.05,
                          boosting_type= 'gbdt',
                          subsample = 0.8,
                          n_estimators=500,
                          max_depth=3,
                          random_state=42,
                          silent=True,
                          n_jobs=-1,
                          )
clf.fit(X_train, y_train)

ax = lgbm_plt(clf,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

print('Accuracy on training set: {:.2f}'
       .format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'
       .format(clf.score(X_test[X_train.columns], y_test)))

# # Light GBM
# precision: 0.797
# recall: 0.838
# f1-score: 0.817
# accuracy: 0.813
# Accuracy on training set: 0.84
# Accuracy on test set: 0.81

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2),
 'subsample':np.arange(0.5,1,0.1),
 'alpha':[0,1],
 'lambda':[0,1]
}
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(), 
param_grid = param_test1, scoring='f1',n_jobs=-1, cv=2, verbose=3)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_
# ({'alpha': 1,
#   'lambda': 1,
#   'max_depth': 5,
#   'min_child_weight': 3,
#   'subsample': 0.5},
#  0.8069929442456798)

# """TUNE"""
clf2 = lgb.LGBMClassifier(objective=' y',
                          metric='logloss',
                          learning_rate = 0.05,
                          boosting_type= 'gbdt',
                          subsample = 0.5,
                          n_estimators=500,
                          max_depth=5,
                          random_state=42,
                          silent=True,
                          n_jobs=-1,
                          )
clf2.fit(X_train, y_train)

ax = lgbm_plt(clf2,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred=clf2.predict(X_test)
y_pred_train = clf2.predict(X_train)

print('Light GBM')
print('precision: ' + "{:.3f}".format(precision_score(y_test, y_pred)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, y_pred)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, y_pred)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, y_pred)))

print('Accuracy on training set: {:.2f}'
       .format(clf2.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'
       .format(clf2.score(X_test[X_train.columns], y_test)))



pickle.dump(clf2,open('spaceship_titanic_saved_model.pkl','wb'))
clf_f = pickle.load(open('spaceship_titanic_saved_model.pkl','rb'))


pred = clf2.predict(df2_test)
out = pd.DataFrame({'PassengerId':df2_test.index, 'Transported':pred})

inv_map = {v: k for k, v in mapping.items()}
out['Transported'] = out['Transported'].map(inv_map)

out.to_csv('spaceship_titanic_pred.csv',index=False)

