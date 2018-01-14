# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:18:52 2018

@author: Kirill
"""

from contextlib import redirect_stdout

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from sklearn import preprocessing
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


negative_list = ['give me a break',
 'bitch',
 'fuck off',
 'son of a bitch',
 'fucking shit',
 'dumb',
 'what the fucking hell',
 'jerk',
 'poop',
 'prick',
 'sucking',
 'fucking idiot',
 'faggot',
 'assholes',
 'loser',
 'homo',
 'sucked',
 'fuck up',
 'asscrack',
 'grammar nazi',
 'crap',
 'retard',
 'ass',
 'stupid',
 'coz',
 'slut',
 'stupid',
 'dick',
 'asses',
 'fucked',
 'shit',
 'asshole',
 'nerd',
 'idiot',
 'no way',
 'sucks',
 'fuck around',
 'gay',
 'motherfucker',
 "fuck",
 'fucking',
 'fool',
 'nigger',
 'freak',
 'fucking',
 'noob',
 'hooker',
 'bumfuck',
 'cunt',
 'kiss my ass',
 'douchebag',
 'fuck you',
 'innit',
 'whore',
 'damn',
 'dafaq',
 'fool',
 'shit',
 'fucking',
 'bastard',
 'pussy',
 'are you nuts',
 'ass',
 'fuck',
 'bullshit',
 'sucker',
 'bastards',
 'screw it',
 'fuck with']


politeness_list = ['thanlful',
 'pardon',
 'excuse',
 'kindfavor',
 'blessing',
 'thanx',
 'could',
 'kind',
 'hope',
 'anyone',
 'thankful',
 'help',
 'course',
 'certainly',
 'thakyou',
 'thankfully',
 'sorry',
 'slight',
 'advance',
 'thank you',
 'thx',
 'slightly',
 'wonderful',
 'hope',
 'reading',
 'perhaps',
 'anything',
 'would',
 'bless',
 'seem',
 'grateful',
 'thank',
 'bit',
 'everyone',
 'blessed',
 'excuse',
 'wonderfully',
 'blessings',
 'thanks',
 'blessins',
 'thankyou']


reciprocity_list = ['share the love',
 'repay in kind',
 'return a favor',
 'return kindness',
 'repay the kindness',
 'pay it forward',
 'share kindness']


driver_list = [
 'burger',
 'burgers',
 'yogurt',
 'someone',
 'couple',
 'help',
 'rice',
 'luck',
 'anything',
 'ramen',
 'today',
 'food',
 'school',
 'pizza',
 'favor',
 'end',
 'stamp',
 'grocery',
 'vegetables',
 'vegetable',
 'vegitarian',
 'vegetarians',
 'vegans',
 'vegan',
 'house',
 'anyone']


family_list = ['wife',
 'boyfriend',
 'boyfriendly',
 'boyfriends',
 'boy',
 'boys',
 'brother',
 'brothers',
 'dad',
 'mum',
 'husband',
 'parent',
 'son',
 'mother',
 'daughter',
 'mom',
 'family',
 'father',
 'parents']


money_list = ['bills',
 'check',
 'buy',
 'budget',
 'cash',
 'payed',
 'budgeting',
 'ﬁnancial',
 'spent',
 'paycheck',
 'budgeted',
 'credit',
 'current',
 'money',
 'bucks',
 'bill',
 'unpaid',
 'rent',
 'loan',
 'deposit',
 'poor',
 'visa',
 'account',
 'due',
 'dollar',
 'paid',
 'broke',
 'bank',
 'still',
 'usd',
 'dollars']


time_list = ['tonight',
 'time',
 'week',
 'month',
 'Friday',
 'today',
 'day',
 'now',
 'after',
 'till',
 'evening',
 'when',
 'year',
 'tomorrow',
 'tonight',
 'hour',
 'years',
 'before',
 'long',
 'until',
 'yesterday',
 'morning',
 'soon',
 'past',
 'ﬁrst',
 'while',
 'never',
 'next',
 'last',
 'ago',
 'night']


job_list = ['employment',
 'ﬁred',
 'job',
 'interview',
 'work',
 'hired',
 'hire',
 'unemployment']


student_list = ['studying',
 'project',
 'study',
 'tuition',
 'student',
 'ﬁnals',
 'college',
 'semester',
 'university',
 'class',
 'school',
 'roommate',
 'dorm']


craving_list = ['movie',
 'celebrate',
 'games',
 'beer',
 'drunk',
 'celebrating',
 'friend',
 'crave',
 'invite',
 'boyfriend',
 'invited',
 'girlfriend',
 'date',
 'drinks',
 'birthday',
 'wasted',
 'craving',
 'party',
 'game']


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
        for word in train_df.requester_subreddits_at_request[ind]:
            if word in neg_count:
                neg_count[word] += 1
            else:
                neg_count[word] = 1
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
        return 0


def photos(raw):
    return raw.count('http://i.imgur.com/')


def edited_text(raw):
    word_list = raw.split()
    if 'EDIT' in word_list:
        return 1
    else:
        return 0


def text_produce(raw, terms_list):
    result = 0
    word_list = raw.lower()
    for item in set(terms_list):
        if item in word_list:
            result += 1
    if result > 0:
            return result
    else:
        return 0


def classifiers_evaluation(df_res, y):

    classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(),
    SGDClassifier(loss='log', max_iter=1000),
    LogisticRegressionCV()]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

#    y = df_res['requester_received_pizza']
#    del df_res['requester_received_pizza']
#    X = df_res
    quantile = preprocessing.QuantileTransformer()
    X = quantile.fit_transform(df_res)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
#            acc = accuracy_score(y_test, train_predictions)
            acc = roc_auc_score(y_test, train_predictions)


            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    print(acc_dict)
    print(log)

#    plt.xlabel('ROC_AUC score')
#    plt.title('Classifier ROC_AUC score')
#    sns.set(rc={'figure.figsize':(18,12)})
#    sns.set_color_codes("muted")
#    sns.barplot(x='ROC_AUC score', y='Classifier', data=log, color="b")


def feature_selection(df_res, y, file_number):

    feature_names = df_res.columns

#    min_max_scaler = preprocessing.MinMaxScaler()
#    df_res = min_max_scaler.fit_transform(df_res)
    quantile = preprocessing.QuantileTransformer()
    df_res = quantile.fit_transform(df_res)

    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(df_res, y, test_size = 0.25, random_state = 11, shuffle=True)

    gbc = ensemble.GradientBoostingClassifier()#n_estimators=100, random_state=11, n_jobs=-1
    gbc.fit(X_res_train, y_res_train)
    err_train = np.mean(y_res_train != gbc.predict(X_res_train))
    err_test = np.mean(y_res_test != gbc.predict(X_res_test))
    print(err_train, err_test)
    scores = cross_val_score(gbc, df_res, y, cv=5, scoring='roc_auc')
#    print(scores)
    print("ROC_AUC GradientBoostingClassifier: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std()))

#    feature_names = df_res.columns
    importances = gbc.feature_importances_
    indices = np.argsort(importances)[::-1]

    low_cost_features = list()
    with open('res_' + str(file_number) + '.txt', 'w') as f:
        with redirect_stdout(f):
#            print("Feature importances:")
            for f, idx in enumerate(indices):
#                print("{:2d}. feature '{:5s}' ({:.12f})".format(f + 1, feature_names[idx], importances[idx]))
                if importances[idx] == 0:
                    low_cost_features.append(feature_names[idx])
    print('Кол-во пустых фич: ', len(low_cost_features))
    return low_cost_features


def soft_voting(df_res, y):
    print('\n')
    print('SOFT VOTING')

#    min_max_scaler = preprocessing.MinMaxScaler()
#    df_res = min_max_scaler.fit_transform(df_res)

#    robust_scaler = preprocessing.RobustScaler()
#    df_res = robust_scaler.fit_transform(df_res)

    quantile = preprocessing.QuantileTransformer()
    df_res = quantile.fit_transform(df_res)

    clf1 = ensemble.AdaBoostClassifier()
    clf2 = MLPClassifier()#AdaBoostClassifier()#ensemble.RandomForestClassifier(n_estimators=200, random_state=11,n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier()#ensemble.GradientBoostingClassifier(n_estimators=3000, learning_rate=1.1, max_depth=5, random_state=11)
    clf4 = SGDClassifier(loss='log', max_iter=1000)#SGDClassifier(max_iter=35000, tol=1e-4, shuffle=True, penalty='l2', loss='log')
    clf5 = LogisticRegression()
    clf6 = LogisticRegressionCV()
    clf7 = QuadraticDiscriminantAnalysis()
    clf8 = GaussianNB()
    clf9 = LinearDiscriminantAnalysis()
    eclf = VotingClassifier(estimators=[('ada', clf1), ('mlpc', clf2),
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('lgr', clf5), ('lrcv', clf6),
                                        ('qda', clf7), ('gnb', clf8),
                                        ('lda', clf9)],
                                voting='soft', weights=[1,1,1,1,1,1,1,1,1])

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, eclf],
                          ['AdaBoostClassifier', 'MLPClassifier',
                           'GradientBoosting', 'SGDClassifier',
                           'LogisticRegression', 'LogisticRegressionCV',
                           'QuadraticDiscriminantAnalysis', 'GaussianNB',
                           'LinearDiscriminantAnalysis', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, scoring='roc_auc') #, scoring='roc_auc'
        print("ROC_AUC scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))


def hard_voting(df_res, y):

    print('HARD VOTING')

    quantile = preprocessing.QuantileTransformer()
    df_res = quantile.fit_transform(df_res)

    clf1 = ensemble.AdaBoostClassifier()
    clf2 = MLPClassifier()#AdaBoostClassifier()#ensemble.RandomForestClassifier(n_estimators=200, random_state=11,n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier()#ensemble.GradientBoostingClassifier(n_estimators=3000, learning_rate=1.1, max_depth=5, random_state=11)
    clf4 = SGDClassifier(loss='log', max_iter=1000)#SGDClassifier(max_iter=35000, tol=1e-4, shuffle=True, penalty='l2', loss='log')
    clf5 = LogisticRegression()
    clf6 = LogisticRegressionCV()
    clf7 = QuadraticDiscriminantAnalysis()
    clf8 = GaussianNB()
    clf9 = LinearDiscriminantAnalysis()
    eclf = VotingClassifier(estimators=[('ada', clf1), ('mlpc', clf2),
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('lgr', clf5), ('lrcv', clf6),
                                        ('qda', clf7), ('gnb', clf8),
                                        ('lda', clf9)],
                                voting='hard')

#    eclf = VotingClassifier(estimators=[('ada', clf1), ('mlpc', clf2),
#                                        ('gbs', clf3), ('sgdc', clf4), ('lgr', clf5), ('lrcv', clf6)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, eclf],
                          ['AdaBoostClassifier', 'MLPClassifier',
                           'GradientBoosting', 'SGDClassifier',
                           'LogisticRegression', 'LogisticRegressionCV',
                           'QuadraticDiscriminantAnalysis', 'GaussianNB',
                           'LinearDiscriminantAnalysis', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, scoring='roc_auc') #, scoring='roc_auc'
        print("ROC_AUC scoring: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))


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
#Группы в которых находятся только получатели пиццы в обучающей выборке
    positive_result_groups = unique_group(train_df)
    print('Уникальных групп на сайте у получателей пиццы', len(positive_result_groups))

    train_df.loc[train_df.giver_username_if_known == 'N/A', 'giver_username_if_known'] = 0
    train_df.loc[train_df.giver_username_if_known != 0, 'giver_username_if_known'] = 1
    train_df['giver_username_if_known'] = pd.to_numeric(train_df['giver_username_if_known'], errors='coerse')

    train_df['time_delta'] = train_df['unix_timestamp_of_request'] - train_df['unix_timestamp_of_request_utc']

    remove = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']

    for rem in remove:
        del train_df[rem]

#Группы в которых находятся только получатели пиццы в обучающей выборке
    for gr in positive_result_groups:
        train_df[gr] = 0
    for ind in train_df[train_df.requester_received_pizza == 1].index:
        for word in train_df.requester_subreddits_at_request[ind]:
            if word in positive_result_groups:
                train_df.at[ind, word] = 1

    train_df['text_length'] = train_df['request_text_edit_aware'].apply(lambda x: text_length(x))
    train_df['title_length'] = train_df['request_title'].apply(lambda x: text_length(x))
    train_df['photos_number'] = train_df['request_text_edit_aware'].apply(lambda x: photos(x))
    train_df['text_was_edited'] = train_df['request_text_edit_aware'].apply(lambda x: edited_text(x))
    train_df['negative_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, negative_list))
    train_df['politeness_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, politeness_list))
    train_df['reciprocity_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, reciprocity_list))
    train_df['driver_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, driver_list))
    train_df['job_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, job_list))
    train_df['money_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, money_list))
    train_df['student_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, student_list))
    train_df['craving_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, craving_list))
    train_df['time_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, time_list))
    train_df['family_words'] = train_df['request_text_edit_aware'].apply(lambda x: text_produce(x, family_list))

    numerical_columns   = [c for c in train_df.columns if train_df[c].dtype.name != 'object']
    df_res = train_df[numerical_columns]

    y = df_res['requester_received_pizza']
    del df_res['requester_received_pizza']
    print('\n')
    print('Результаты на ПОЛНЫХ данных')
    print('\n')
    classifiers_evaluation(df_res, y)
    soft_voting(df_res, y)

#    poly = preprocessing.PolynomialFeatures(degree=1, interaction_only=True)
#    poly.fit(df_res)
#    df_res = poly.transform(df_res)
#    df_res = np.delete(df_res, np.s_[0:1], axis=1)

#    min_max_scaler = preprocessing.MinMaxScaler()
#    df_res = min_max_scaler.fit_transform(df_res)

    file_number = 0
    low_cost_features = feature_selection(df_res, y, file_number)

    while low_cost_features:
        file_number += 1
        for feature in low_cost_features:
            del df_res[feature]
        low_cost_features = feature_selection(df_res, y, file_number)

    print('\n')
    print('Результаты на СОКРАЩЕННЫХ данных')
    print('\n')

    classifiers_evaluation(df_res, y)

    soft_voting(df_res, y)

#    hard_voting(df_res, y)





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