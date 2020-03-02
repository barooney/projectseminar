# -*- coding: utf-8 -*-

# imports

# standard library
import argparse
import sys
import math
from collections import Counter
import tqdm

# third party modules
import pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np




def fill_bins(STATE, states_to_fetch_reviews):
    ''' bin notation: lower limit excluded and upper limit included'''
    dict_dfs_loot_states = {loot_state : pd.read_json('./data/intermediate/' + loot_state + '_reviews' + '.json', lines=True) for loot_state in states_to_fetch_reviews}   
    df_basis = pd.read_json('./data/intermediate/' + STATE + '_reviews' + '.json', lines=True)
    print("\n\nshape of illinois df before filling: ", df_basis.shape)
    max_funny = df_basis['funny'].max()
    bins =  [-1,0,1,4,max_funny]
    df_basis['funniness_category'] = pd.cut(df_basis.funny, bins, labels=[1,2,3,4])
    # print("type:")
    # print(type(df_basis.query('funniness_category==2').shape[0]))
    
    for each_loot_state in tqdm.tqdm(dict_dfs_loot_states.values()):
        max_funny = each_loot_state['funny'].max()
        bins =  [-1,0,1,4,max_funny]
        each_loot_state['funniness_category'] = pd.cut(df_basis.funny, bins, labels=[1,2,3,4])
    
    
    counter_obj = Counter(df_basis['funniness_category'].tolist())
    print("before filling: ", counter_obj)
    max_reviews_in_highest_bin = counter_obj.most_common(1)[0][1]
    
    for bin_number, num_reviews in tqdm.tqdm(counter_obj.items()):
        if bin_number != counter_obj.most_common(1)[0][0]: # don't fill the bin with reviews that has aleady the most reviews
            for dataframe_state in dict_dfs_loot_states.values():
                if df_basis.query('funniness_category==@bin_number').shape[0] < max_reviews_in_highest_bin:
                    # fill bin with reviews from other states as long as it's not as full as the bin with the most reviews
                    loot_state_corresponding_bin_number = dataframe_state.query('funniness_category==@bin_number')
                    df_basis = pd.concat([df_basis, loot_state_corresponding_bin_number], ignore_index=True)
                else:
                    break
    
    print("\nshape of illinois df after filling: ", df_basis.shape)
    print("bins: ", bins)
    print("after filling: ", Counter(df_basis['funniness_category'].tolist()))
    #return dict_dfs_loot_states



fill_bins("Illinois", ["Wisconsin", "Ohio", "Nevada", "Quebec", "Pennsylvania", "Ontario", "Alberta", "Arizona"])
        
    
    