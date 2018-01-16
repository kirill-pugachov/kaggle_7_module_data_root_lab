# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:59:05 2018

@author: Kirill
"""

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import DBSCAN


data_path = 'Data//'
train_data = 'train.json'
test_data = 'test.json'

start = 1
end = 3

def get_data(data_path, train_data):
    return data_path + train_data

    
def make_frame(train_url):
    df = pd.read_json(train_url)
    return df


def creat_new_stop(train_df):
    stop = stopwords.words('english') + ['next']
    vect = TfidfVectorizer(stop_words=stop)
    vect.fit(train_df['request_text_edit_aware'])
    word_dict = vect.vocabulary_
    
    garb_list = list()
    res_list = list()
    for word in word_dict.keys():
        if len(word) > 2:
            if not word.isdigit():
                if word.isalnum():
                    res_list.append(word)
                else:
                   garb_list.append(word) 
            else:
                garb_list.append(word)
        else:
            garb_list.append(word)
    
    stop = stop + garb_list
    return stop


def text_getter(train_df, k):
    bag_of_words = list()
    for ind in train_df[train_df.requester_received_pizza == k].index:
        bag_of_words.append(train_df.request_text_edit_aware[ind])
    return bag_of_words
    
    
def vectorizer(start, end, stop_w):

    mass_vectorizer = TfidfVectorizer(
            ngram_range=(start, end),
            token_pattern=r'\b\w\w\w+\b',
            stop_words=stop_w,
            analyzer='word',
            use_idf=False)
    
    return mass_vectorizer    


def words_to_bag(positive_bag, stop_w):
    vect = vectorizer(start, end, stop_w)
    vect.fit(positive_bag)
    res_dict = dict()
    for line in positive_bag:
        c_9 = (list(zip(list(vect.get_feature_names()), list(vect.transform([line]).toarray()[0]))))
        for c in c_9:
#            print(c)
            if c[0] in res_dict:
#                print([0])
                res_dict[c[0]] += c[1]
            else:
#                print(c[1])
                res_dict[c[0]] = c[1]
    return res_dict


def key_list_dict(train_df, k):
    new_stop_words = creat_new_stop(train_df)
    positive_bag = text_getter(train_df, k)
    positive_result = words_to_bag(positive_bag, new_stop_words)
    return positive_result


def separated_semantic_df(positive_result, negative_result):
    pos_res = set(positive_result.keys()).difference(set(negative_result.keys()))
    positive = dict()
    for pos in pos_res:
        positive[pos] = positive_result[pos]
    positive_df = pd.DataFrame.from_dict(positive, orient='index', dtype=None)  
    return positive_df


def clean_semantic(res_df):
    return res_df[res_df[0] > res_df[0].quantile(0.75)]
   

def words_feature_counter(train_df, pos_df, new_title):
    train_df[new_title] = 0
    import re
    pattern = r'\b\w\w\w+\b'
    res_dict = dict()
    for ind in train_df.index:
        res = [0, 0]
        for text in pos_df.index:
            if text in ' '.join(re.findall(pattern, train_df['request_text_edit_aware'][ind])):
                res[0] += 1
            else:
                res[1] += 1
#        print(res)
        res_dict[ind] = res
        train_df.loc[ind, new_title] = res[0]
    return train_df
   
if __name__ == '__main__':
    
    train_url = get_data(data_path, train_data)
    train_df = make_frame(train_url)
    
    positive_result = key_list_dict(train_df, 1)
    negative_result = key_list_dict(train_df, 0)
    
    positive_df = separated_semantic_df(positive_result, negative_result)
    negative_df = separated_semantic_df(negative_result, positive_result)
    
    pos_df = clean_semantic(positive_df)
    neg_df = clean_semantic(negative_df)
    
    new_title = 'positive_words'
    train_df = words_feature_counter(train_df, pos_df, new_title)
    
    new_title = 'negative_words'
    train_df = words_feature_counter(train_df, neg_df, new_title)
    

#def final_process(data_path, train_data):
#    train_url = get_data(data_path, train_data)
#    train_df = make_frame(train_url)
#    
#    positive_result = key_list_dict(train_df, 1)
#    negative_result = key_list_dict(train_df, 0)
#    
#    positive_df = separated_semantic_df(positive_result, negative_result)
#    negative_df = separated_semantic_df(negative_result, positive_result)
#    
#    pos_df = clean_semantic(positive_df)
#    neg_df = clean_semantic(negative_df)
#    return pos_df, neg_df
#
#    
#    pos, neg = final_process(data_path, train_data)
#    print('Positive words dataframe: ', type(pos), pos.shape())
#    print('Negative words dataframe: ', type(neg), neg.shape())    
#    pos_res = set(positive_result.keys()).difference(set(negative_result.keys()))
#    neg_res = set(negative_result.keys()).difference(set(positive_result.keys()))
#    
#    positive = dict()
#    for pos in pos_res:
#        positive[pos] = positive_result[pos]
#    
#    negative = dict()
#    for neg in neg_res:
#        negative[neg] = negative_result[neg]
#        
#    
#    positive_df = pd.DataFrame.from_dict(positive_result, orient='index', dtype=None)
#    negative_df = pd.DataFrame.from_dict(negative_result, orient='index', dtype=None)







    
#    new_stop_words = creat_new_stop(train_df)
#    
#    positive_bag = text_getter(train_df, 1)
#    negative_bag = text_getter(train_df, 0)
#    
#    positive_result = words_to_bag(positive_bag, new_stop_words)
#    negative_result = words_to_bag(negative_bag, new_stop_words)
#    
#    positive = positive_result.keys() - negative_result.keys()
#    negative = negative_result.keys() - positive_result.keys()




#    stop = stopwords.words('english') + ['next']
#    vect = TfidfVectorizer(stop_words=stop)
#    vect.fit(train_df['request_text_edit_aware'])
#    word_dict = vect.vocabulary_
#    
#    items = [(v, k) for k, v in word_dict.items()]
#    items.sort()
#    items.reverse()
#    word_dict = dict([(k, v) for v, k in items])
#    print(len(word_dict.keys()))
#    
#    garb_list = list()
#    res_list = list()
#    for word in word_dict.keys():
#        if len(word) > 2:
#            res_list.append(word)
#        else:
#            garb_list.append(word)
#    
#    stop = stop + garb_list
#    vect = TfidfVectorizer(stop_words=stop)
#    vect.fit(train_df['request_text_edit_aware'])
#    word_dict_1 = (list(vect.vocabulary_.keys())).sort()
    
#    X = vect.transform(word_dict_1.keys())
#    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            