# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:18:52 2018

@author: Kirill
"""

import pandas as pd
import numpy as np
import json
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from pandas.plotting import scatter_matrix
from contextlib import redirect_stdout


data_path = 'Data//'
train_data = 'train.json'
test_data = 'test.json'

def get_data(data_path, train_data):
    return data_path + train_data

    
def make_frame(train_url):
    df = pd.read_json(train_url)
    return df


def group_selector(train_df, k=0):
    neg_bag_of_words = list()
    neg_count = dict()
    for ind in train_df[train_df.requester_received_pizza == k].index:
        neg_bag_of_words += train_df.requester_subreddits_at_request[ind]
#        print(ind, len(train_df.requester_subreddits_at_request[ind]))
        for word in train_df.requester_subreddits_at_request[ind]:
            if word in neg_count:
                neg_count[word] += 1
            else:
                neg_count[word] = 1
#    print(len(neg_bag_of_words), len(set(neg_bag_of_words)))
    return neg_count


def unique_group(train_df):
    neg_count = group_selector(train_df, k=0)
    res_count = group_selector(train_df, k=1)
    unique_group = res_count.keys() - neg_count.keys()   
    return unique_group


def text_length(raw):
    return len(raw)

def photos(raw):
    return raw.count('http://i.imgur.com/')

def model_educate(df_res, y):
    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.3, random_state = None)    
    rf = ensemble.RandomForestClassifier(n_estimators=300, random_state=None)
    rf.fit(X_res_train, y_res_train)
    err_train = np.mean(y_res_train != rf.predict(X_res_train))
    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
    print(err_train, err_test)    
    scores = cross_val_score(rf, df_res, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    
    feature_names = df_res.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]              
    
    low_cost_features = list()
    with open('res_0.txt', 'w') as f:
        with redirect_stdout(f):
            print("Feature importances:")
            for f, idx in enumerate(indices):
                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
                if importances[idx] == 0:
                    low_cost_features.append(feature_names[idx])
    print('Кол-во пустых фич: ', len(low_cost_features))
    return low_cost_features
    
if __name__ == '__main__':
    train_url = get_data(data_path, train_data)
    train_df = make_frame(train_url)
   

    test_url = get_data(data_path, test_data)
    test_df = make_frame(test_url)


    train_df.loc[train_df.post_was_edited > 1, 'post_was_edited'] = 1
    train_df['requester_received_pizza'] = (train_df['requester_received_pizza'] == True).astype(int)
    categorical_columns = [c for c in train_df.columns if train_df[c].dtype.name == 'object']
    numerical_columns   = [c for c in train_df.columns if train_df[c].dtype.name != 'object']
#    print('Категориальные колонки фрейма: ', '\n', categorical_columns)
#    print('Цифровые колонки фрейма: ', '\n', numerical_columns)
    train_df[numerical_columns].describe()
    data_nonbinary = pd.get_dummies(train_df['requester_user_flair'])
    df_res = pd.concat((train_df[numerical_columns], data_nonbinary), axis=1)
#    df_res = train_df[numerical_columns]
    remove = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']
    for rem in remove:
        del df_res[rem]
    y = df_res['requester_received_pizza']
    del df_res['requester_received_pizza']
    df_res = (df_res - df_res.mean()) / df_res.std()
    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.3, random_state = None)
#    y_res_train = X_res_train['requester_received_pizza']
#    del X_res_train['requester_received_pizza']
#    y_res_test = X_res_test['requester_received_pizza']
#    del X_res_test['requester_received_pizza']
    rf = ensemble.RandomForestClassifier(n_estimators=300, random_state=None)
    rf.fit(X_res_train, y_res_train)
    err_train = np.mean(y_res_train != rf.predict(X_res_train))
    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
    print(err_train, err_test)
    scores = cross_val_score(rf, df_res, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    
    feature_names = df_res.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature importances:")
    for f, idx in enumerate(indices):
        print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
    
    print('\n')
    print('Второй заход на задачу с учетом вида тестовых данных', '\n', '\t', 'Читай всю доку по проекту, бля*ь!!!', '\n')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Второй заход на задачу с учетом вида тестовых данных 
#Читай всю доку по проекту, бля*ь!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    train_url = get_data(data_path, train_data)
    train_df = make_frame(train_url)
   
    test_url = get_data(data_path, test_data)
    test_df = make_frame(test_url)
    
    for col in train_df.columns:
        if col != 'requester_received_pizza':
            if col not in test_df.columns:
                del train_df[col]
        
    positive_result_groups = unique_group(train_df)
    print('Уникальных групп на сайте у получателей пиццы', len(positive_result_groups))
    
    train_df.loc[train_df.giver_username_if_known == 'N/A', 'giver_username_if_known'] = 0
    train_df.loc[train_df.giver_username_if_known != 0, 'giver_username_if_known'] = 1
    train_df['giver_username_if_known'] = pd.to_numeric(train_df['giver_username_if_known'], errors='coerse')
    
    train_df['time_delta'] = train_df['unix_timestamp_of_request'] - train_df['unix_timestamp_of_request_utc']
    
    remove = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']
    
    for rem in remove:
        del train_df[rem]
    
    for gr in positive_result_groups:
        train_df[gr] = 0
    for ind in train_df[train_df.requester_received_pizza == 1].index:
        for word in train_df.requester_subreddits_at_request[ind]:
            if word in positive_result_groups:
                train_df.at[ind, word] = 1
    
    train_df['text_length'] = train_df['request_text_edit_aware'].apply(lambda x: text_length(x))
    train_df['title_length'] = train_df['request_title'].apply(lambda x: text_length(x))
    train_df['photos_number'] = train_df['request_text_edit_aware'].apply(lambda x: photos(x))
    
    numerical_columns   = [c for c in train_df.columns if train_df[c].dtype.name != 'object']
    df_res = train_df[numerical_columns]
    
    
    y = df_res['requester_received_pizza']
    del df_res['requester_received_pizza']
    
    df_res = (df_res - df_res.mean()) / df_res.std()
    
    low_cost_features = model_educate(df_res, y)
    
    while low_cost_features:
        for feature in low_cost_features:
            del df_res[feature]
        low_cost_features = model_educate(df_res, y)
    
##    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.3, random_state = None)    
##
##    rf = ensemble.RandomForestClassifier(n_estimators=300, random_state=None)
##    rf.fit(X_res_train, y_res_train)
##    err_train = np.mean(y_res_train != rf.predict(X_res_train))
##    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
##    print(err_train, err_test)    
##    scores = cross_val_score(rf, df_res, y, cv=5)
##    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
##    
##    feature_names = df_res.columns
##    importances = rf.feature_importances_
##    indices = np.argsort(importances)[::-1]              
##    
##    low_cost_features = list()
##    with open('res_0.txt', 'w') as f:
##        with redirect_stdout(f):
##            print("Feature importances:")
##            for f, idx in enumerate(indices):
##                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
##                if importances[idx] == 0:
##                    low_cost_features.append(feature_names[idx])
##    print('Кол-во пустых фич: ', len(low_cost_features))
#    
#    for feature in low_cost_features:
#        del df_res[feature]
#    
#    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.3, random_state = None)    
#
#    rf = ensemble.RandomForestClassifier(n_estimators=300, random_state=None)
#    rf.fit(X_res_train, y_res_train)
#    err_train = np.mean(y_res_train != rf.predict(X_res_train))
#    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
#    print(err_train, err_test)    
#    scores = cross_val_score(rf, df_res, y, cv=5)
#    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#    
#    feature_names = df_res.columns
#    importances = rf.feature_importances_
#    indices = np.argsort(importances)[::-1]              
#    
#    low_cost_features = list()
#    with open('res_1.txt', 'w') as f:
#        with redirect_stdout(f):
#            print("Feature importances:")
#            for f, idx in enumerate(indices):
#                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
#                if importances[idx] == 0:
#                    low_cost_features.append(feature_names[idx])
#    print('Кол-во пустых фич: ', len(low_cost_features))
#
#    for feature in low_cost_features:
#        del df_res[feature]
#    
#        X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.3, random_state = None)    
#
#    rf = ensemble.RandomForestClassifier(n_estimators=300, random_state=None)
#    rf.fit(X_res_train, y_res_train)
#    err_train = np.mean(y_res_train != rf.predict(X_res_train))
#    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
#    print(err_train, err_test)    
#    scores = cross_val_score(rf, df_res, y, cv=5)
#    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#    
#    feature_names = df_res.columns
#    importances = rf.feature_importances_
#    indices = np.argsort(importances)[::-1]              
#    
#    low_cost_features = list()
#    with open('res_2.txt', 'w') as f:
#        with redirect_stdout(f):
#            print("Feature importances:")
#            for f, idx in enumerate(indices):
#                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
#                if importances[idx] == 0:
#                    low_cost_features.append(feature_names[idx])
#    print('Кол-во пустых фич: ', len(low_cost_features))                
#    
    
#    pca = decomposition.PCA(n_components=21)
#    pca.fit(df_res)
#    X = pca.transform(df_res)
#    train_df['requester_received_pizza'] = pd.to_numeric(train_df['requester_received_pizza'], errors='coerse', inplace=True)
#    train_df.at[train_df['requester_received_pizza'] == False, 'requester_received_pizza'] = 0
#    train_df.at[train_df['requester_received_pizza'] == True, 'requester_received_pizza'] = 1