# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:19:02 2018

@author: Kirill
"""

import pandas as pd
import numpy as np
#import json
#from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from pandas.plotting import scatter_matrix
from contextlib import redirect_stdout
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB


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
    pos_unique_group = res_count.keys() - neg_count.keys()
    neg_unique_group = neg_count.keys() - res_count.keys()
    return pos_unique_group, neg_unique_group


def group_popularity(train_df, positive_result_groups, k):
    res_dict = dict()
    for ind in train_df[train_df.requester_received_pizza == k].index:
        for group in train_df.requester_subreddits_at_request[ind]:
            if group in positive_result_groups:
                if group in res_dict:
                    res_dict[group] += 1
                else:
                    res_dict[group] = 1
    return res_dict
                      

def group_filter(pos_pop_group):
    pos_df = pd.DataFrame.from_dict(pos_pop_group, orient='index', dtype=None)
    return pos_df[pos_df[0] > 1]

    
if __name__ == '__main__':
    
    train_url = get_data(data_path, train_data)
    train_df = make_frame(train_url)

#Группы в которых находятся только получатели пиццы в обучающей выборке        
    positive_result_groups, negative_result_groups = unique_group(train_df)
    
#популярность групп
    pos_pop_group = group_popularity(train_df, positive_result_groups, 1)
    neg_pop_group = group_popularity(train_df, negative_result_groups, 0)
    
#негативные и позитивные групп > 1 участника из нег и поз получателей
    pos_df = group_filter(pos_pop_group)
    neg_df = group_filter(neg_pop_group)
    
    