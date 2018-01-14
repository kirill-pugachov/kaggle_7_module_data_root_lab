# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:18:52 2018

@author: Kirill
"""

import pandas as pd
import numpy as np
#import json
#from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from pandas.plotting import scatter_matrix
from contextlib import redirect_stdout
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB


negative_list = ['shit', 'ass', 'crap', 'fuck', 'fuck around', 'fuck off', 
                 'fuck up', 'fuck with', 'fuck you', 'fucked', 'fucking', 
                 'fucking ass', 'fucking fool', 'fucking idiot', 'idiot',
                 'fucking shit', 'damn', 'nigger', 'whore', 'slut', 
                 'bitch', 'freak', 'douchebag', 'gay', 'faggot', 'homo', 
                 'bastard', 'asshole', 'jerk', 'prick', 'dick', 'cunt', 
                 'pussy', 'loser', 'sucker', 'nerd', 'noob', 'fool', 
                 'stupid', 'dumb', 'retard', 'grammar nazi',
                 'are you nuts', 'cut the bullshit', 
                 'i\'m gonna kick your ass', 'dafaq', 'give me a break', 
                 'no way', 'innit', 'kiss my ass', 'i don\'t give a shit',  
                 'i don\'t give a fuck', 'coz', 'move your ass', 'motherfucker', 
                 'hooker', 'what the fuck do you think you\'re doing', 
                 'what the fucking hell', 'poop', 'you stupid ass', 
                 'screw it', 'son of a bitch']

politeness_list = ['hope', 'seem', 'kind of', 'could', 'would', 'to hope', 
                   'to seem', 'perhaps', 'grateful', 'certainly', 'of course',
                   'beg your pardon', 'excuse me', 'i\'m sorry', 
                   'sorry about that', 'i\'m sorry to', 'sorry for', 
                   'sorry about', 'sorry that', 'grateful', 'thank you', 
                   'thanks', 'thank you for', 'very kind', 'a little', 'a bit', 
                   'a little bit', 'slight', 'slightly', 'thanks', 'advance', 
                   'guy', 'reading', 'anyone', 'pizza', 'anything', 'story', 
                   'tonight', 'help', 'place', 'everyone', 'craving', 'kind'
                   'favor']

reciprocity_list = ['pay it forward', 'share kindness', 'repay in kind', 
                    'share the love', 'return kindness', 'return a favor',
                    'repay the kindness']

drive_words = []

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
    if len(raw.split()) > 1:
        
        return len(raw.split())
    else:
#        print(len(raw.split()), '\t', raw)
        return 0


def photos(raw):
    return raw.count('http://i.imgur.com/')


def edited_text(raw):
    word_list = raw.split()
    if 'EDIT' in word_list:
        return 1
    else:
        return 0


def negative_words(raw):
    result = 0
    word_list = raw.lower().split()
    for negative in set(negative_list):
        if negative in word_list:
            result += 1#raw.lower().count(negative)
#            print(negative)
    if result > 0:
        return 1
    else:
        return 0
        
    
def politeness_words(raw):
    result = 0
    word_list = raw.lower().split()
    for politeness in set(politeness_list):
        if politeness in word_list:
            result += 1#raw.lower().count(negative)
#            print(politeness)
    if result > 0:
        return 1
    else:
        return 0


def reciprocity_text(raw):
    result = 0
    word_list = raw.lower()
    for reciprocity in set(reciprocity_list):
        if reciprocity in word_list:
            result += 1#raw.lower().count(negative)
#            print(politeness)
    if result > 0:
        return 1
    else:
        return 0
    
    
def feature_selection(df_res, y, file_number):
    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.25, random_state = 11, shuffle=True)    
    
    rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11, n_jobs=-1)
    rf.fit(X_res_train, y_res_train)
    err_train = np.mean(y_res_train != rf.predict(X_res_train))
    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
    print(err_train, err_test)    
    scores = cross_val_score(rf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
    print(scores)
    print("ROC_AUC RandomForest: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std()))

    feature_names = df_res.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]              
    
    low_cost_features = list()
    with open('res_' +str(file_number)+ '.txt', 'w') as f:
        with redirect_stdout(f):
            print("Feature importances:")
            for f, idx in enumerate(indices):
                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
                if importances[idx] == 0:
                    low_cost_features.append(feature_names[idx])
    print('Кол-во пустых фич: ', len(low_cost_features))
    
#    gbc = ensemble.GradientBoostingClassifier(n_estimators=1000, 
#                                              learning_rate=1.0, max_depth=6, 
#                                              random_state=11)
#    gbc.fit(X_res_train, y_res_train)
#    err_train = np.mean(y_res_train != gbc.predict(X_res_train))
#    err_test  = np.mean(y_res_test  != gbc.predict(X_res_test))
#    print(err_train, err_test) 
#    scores = cross_val_score(gbc, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
#    print(scores)
#    print("ROC_AUC GradientBoosting: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std()))    
#    
#    sdg = SGDClassifier(max_iter=15000, tol=1e-3, shuffle=True, penalty='l1', loss='perceptron')
#    sdg.fit(X_res_train, y_res_train)
#    err_train = np.mean(y_res_train != sdg.predict(X_res_train))
#    err_test  = np.mean(y_res_test  != sdg.predict(X_res_test))
#    print(err_train, err_test) 
#    scores = cross_val_score(sdg, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
#    print(scores)
#    print("ROC_AUC SGDClassifier: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std())) 
    
    return low_cost_features


def soft_voting(df_res):
#    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, 
#                        y, test_size = 0.25, random_state = 11, shuffle=True)
    clf1 = ensemble.RandomForestClassifier(n_estimators=100, random_state=11, 
                                           n_jobs=-1)
    clf2 = ensemble.GradientBoostingClassifier(n_estimators=1000, 
                                              learning_rate=1.0, max_depth=6, 
                                              random_state=11)
    clf3 = GaussianNB()
    clf4 = SGDClassifier(max_iter=25000, tol=1e-5, shuffle=True, penalty='l2', loss='log')
    eclf = VotingClassifier(estimators=[('rfc', clf1), ('gbs', clf2), 
                                        ('sgdc', clf3), ('gau', clf4)], 
                                voting='soft', weights=[1,0.85,1,0.85])
    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], 
                          ['RandomForestClassifier', 'GradientBoosting', 'GaussianNB', 'SGDClassifier', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        

    
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
    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.25, random_state = 11)
#    y_res_train = X_res_train['requester_received_pizza']
#    del X_res_train['requester_received_pizza']
#    y_res_test = X_res_test['requester_received_pizza']
#    del X_res_test['requester_received_pizza']
    rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
    rf.fit(X_res_train, y_res_train)
    err_train = np.mean(y_res_train != rf.predict(X_res_train))
    err_test  = np.mean(y_res_test  != rf.predict(X_res_test))
    print(err_train, err_test)
    scores = cross_val_score(rf, df_res, y, cv=5, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std()))
    
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
##Группы в которых находятся только получатели пиццы в обучающей выборке        
#    positive_result_groups = unique_group(train_df)
#    print('Уникальных групп на сайте у получателей пиццы', len(positive_result_groups))
    
    train_df.loc[train_df.giver_username_if_known == 'N/A', 'giver_username_if_known'] = 0
    train_df.loc[train_df.giver_username_if_known != 0, 'giver_username_if_known'] = 1
    train_df['giver_username_if_known'] = pd.to_numeric(train_df['giver_username_if_known'], errors='coerse')
    
    train_df['time_delta'] = train_df['unix_timestamp_of_request'] - train_df['unix_timestamp_of_request_utc']
    
    remove = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']
    
    for rem in remove:
        del train_df[rem]

##Группы в которых находятся только получатели пиццы в обучающей выборке       
#    for gr in positive_result_groups:
#        train_df[gr] = 0
#    for ind in train_df[train_df.requester_received_pizza == 1].index:
#        for word in train_df.requester_subreddits_at_request[ind]:
#            if word in positive_result_groups:
#                train_df.at[ind, word] = 1
    
    train_df['text_length'] = train_df['request_text_edit_aware'].apply(lambda x: text_length(x))
    train_df['title_length'] = train_df['request_title'].apply(lambda x: text_length(x))
    train_df['photos_number'] = train_df['request_text_edit_aware'].apply(lambda x: photos(x))
    train_df['negative_words'] = train_df['request_text_edit_aware'].apply(lambda x: negative_words(x))
    train_df['politeness_words'] = train_df['request_text_edit_aware'].apply(lambda x: politeness_words(x))
    train_df['text_was_edited'] = train_df['request_text_edit_aware'].apply(lambda x: edited_text(x))
    train_df['reciprocity_words'] = train_df['request_text_edit_aware'].apply(lambda x: reciprocity_text(x))
    
    numerical_columns   = [c for c in train_df.columns if train_df[c].dtype.name != 'object']
    df_res = train_df[numerical_columns]
    
    
    y = df_res['requester_received_pizza']
    del df_res['requester_received_pizza']
    
    df_res = (df_res - df_res.mean()) / df_res.std()
    
    file_number = 0
    low_cost_features = feature_selection(df_res, y, file_number)
    
    while low_cost_features:
        file_number += 1
        for feature in low_cost_features:
            del df_res[feature]
        low_cost_features = feature_selection(df_res, y, file_number)
    
    soft_voting(df_res)
    
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