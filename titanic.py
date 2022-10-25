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

# import pycaret

import re
import pickle
import os
os.chdir(r"C:\Users\andyhlso\Desktop\Andy\Kaggle\202208 - titanic")

df = pd.read_csv('train.csv')
df2 = df.copy()
df2 = df2.set_index('PassengerId')

drop_vars = ['Name','Ticket']
df2 = df2.drop(drop_vars, axis=1)

df2['Survived'].value_counts()

"""======================================== Data cleaning ========================================"""
# def check_missing_row(missing_rate):   
#     missing_row = (df.isnull().sum(axis=1))/len(df.columns) *100
#     chk = df.loc[missing_row[missing_row >= missing_rate].index]
#     print('Missing rows pct > ' + str(missing_rate) + ' by Group:')
#     print(chk.Group.value_counts())

# def remove_missing_row_data(df, missing_rate):
#     #remove data with 60% missing
#     #most of missing rows are inactive customers, so remove them
#     #check missing value by rows
#     missing_row = (df.isnull().sum(axis=1))/len(df.columns) *100
#     return(df.loc[missing_row[missing_row < missing_rate].index])


def check_missing_col():
    #check missing values by column
    missing = df2.isnull().sum()/ len(df) *100
    missing = missing[missing > 0]
    return(missing)

def remove_missing_col_data(df, missing_rate):
    missing = df2.isnull().sum()/ len(df2) *100
    missing = missing[missing > 0]
    #fill 0 withs missing percentage greater than 80
    missing80pct_cols = missing[missing > missing_rate].index.tolist()
    #df[missing80pct_cols] = df[missing80pct_cols].fillna(0)
    print('no of cols to drop: ' + str(len(missing80pct_cols)))
    return(df.drop(missing80pct_cols, axis=1))

check_missing_col()
# Age         19.865320
# Cabin       77.104377
# Embarked     0.224467

###### DROP CABIN DUE TO HIGH MISSING %
chk = df2['Cabin'].value_counts()
# df2['Cabin_Head'] = df2['Cabin'].str[:1]
# df2['Cabin_Num'] = df2['Cabin'].str.extract('(\d+)')
df2 = df2.drop('Cabin', axis=1)

# df2['Cabin_Head'].value_counts()
# df2['Cabin_Num'].value_counts()

df2['Embarked'].value_counts()
# S    644
# C    168
# Q     77

df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Embarked'] = df2['Embarked'].fillna('S')
# df2['Cabin_Head'].fillna('missing',inplace = True)


one_hot_vars = ['Pclass','Sex','Embarked']
for var in one_hot_vars:
    df2 = pd.get_dummies(df2, columns=[var], prefix = ['OH_' + var])

scale_vars = ['Age','SibSp','Parch','Fare']
df2[scale_vars] = scaler.fit_transform(df2[scale_vars])


corr_df = df2.corr().abs()
df2 = df2.drop('OH_Sex_female', axis = 1)

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
X = df2.drop('Survived', axis=1)
y = df2['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

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

# Light GBM
# precision: 0.806
# recall: 0.730
# f1-score: 0.766
# accuracy: 0.816

# """XGB"""
xg = XGBClassifier().fit(X_train, y_train)
#sorted_idx = np.argsort(xg.feature_importances_)[::-1]

ax = plot_importance(xg,max_num_features = 20)
fig = ax.figure
fig.set_size_inches(10, 10)

y_pred = xg.predict(X_test)
pred_boosting_result = [1 if y_pred[i] >= 0.5 else 0 for i in range(len(y_pred))]

print('xgboost')
print('precision: ' + "{:.3f}".format(precision_score(y_test, pred_boosting_result)))
print('recall: ' + "{:.3f}".format(recall_score(y_test, pred_boosting_result)))
print('f1-score: ' + "{:.3f}".format(f1_score(y_test, pred_boosting_result)))
print('accuracy: ' + "{:.3f}".format(accuracy_score(y_test, pred_boosting_result)))

# xgboost
# precision: 0.743
# recall: 0.743
# f1-score: 0.743
# accuracy: 0.788

top_features = ['Fare','Age','SibSp','Parch','OH_Sex_male']
df3 = df2[top_features]
corr_df3 = df3.corr().abs()

X = df3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

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
# precision: 0.815
# recall: 0.716
# f1-score: 0.763
# accuracy: 0.816

pickle.dump(clf,open('titanic_saved_model.pkl','wb'))
clf_f = pickle.load(open('titanic_saved_model.pkl','rb'))

df_test = pd.read_csv('test.csv')

df_test = df_test.set_index('PassengerId')

df_test = df_test.drop(drop_vars, axis=1)
df_test = df_test.drop('Cabin', axis=1)
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Embarked'] = df_test['Embarked'].fillna('S')
one_hot_vars = ['Pclass','Sex','Embarked']
for var in one_hot_vars:
    df_test = pd.get_dummies(df_test, columns=[var], prefix = ['OH_' + var])

scale_vars = ['Age','SibSp','Parch','Fare']
df_test[scale_vars] = scaler.fit_transform(df_test[scale_vars])
df_test = df_test.drop('OH_Sex_female', axis = 1)
df_test_feat = df_test[['Fare','Age','SibSp','Parch','OH_Sex_male']]

pred = clf.predict(df_test_feat)
out = pd.DataFrame({'PassengerId':df_test_feat.index, 'Survived':pred})
out.to_csv('titanic_pred.csv',index=False)

