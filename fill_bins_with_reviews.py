# -*- coding: utf-8 -*-

# imports

# standard library
import argparse
import sys
import math
from collections import Counter
import tqdm
import itertools

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
    ''' 
    bin notation: lower limit excluded and upper limit included
    only 10 states can be given at a time. With 11 states pandas raises an error claiming that the bins do not raise monotonically.
    '''
    
    # read all states as panda dataframe
    dict_dfs_loot_states = {loot_state.replace(" ", "") : pd.read_json('./data/intermediate/' + loot_state.replace(" ", "") + '_reviews' + '.json', lines=True) for loot_state in states_to_fetch_reviews}   
    df_basis = pd.read_json('./data/intermediate/' + STATE + '_reviews' + '.json', lines=True)
    print("\n\nshape of illinois df before filling: ", df_basis.shape)
      
    # set bins
    max_funny = df_basis['funny'].max()
    bins =  [-1,0,1,2,5,9999999999999999]
    labels=[number for number in range(1,len(bins))]
    
    #### assign bins
    df_basis['funniness_category'] = pd.cut(df_basis.funny, bins, labels=labels)
    # print("type:")
    # print(type(df_basis.query('funniness_category==2').shape[0]))
       
    
    # add funniness category column to each state that's gonna be looted
    for each_loot_state in tqdm.tqdm(dict_dfs_loot_states.values()):
        max_funny = each_loot_state['funny'].max()
        #bins =  [-1,0,1,4,max_funny]
        each_loot_state['funniness_category'] = pd.cut(df_basis.funny, bins=bins, labels=labels)
    
    
    # stats for reviews per bins
    counter_obj = Counter(df_basis['funniness_category'].tolist())
    print("before filling: ", counter_obj)
    max_reviews_in_highest_bin = counter_obj.most_common(1)[0][1]
    print("max reviews in highest bin: " + str(max_reviews_in_highest_bin))
    
    # fill the basis state with reviews from the other states
    for bin_number, num_reviews in tqdm.tqdm(counter_obj.items()):
        if bin_number != counter_obj.most_common(1)[0][0]: # don't fill the bin with reviews that has aleady the most reviews
            for dataframe_state in dict_dfs_loot_states.values():
                # fill bin with reviews from other states as long as it's not as full as the bin with the most reviews
                if df_basis.query('funniness_category==@bin_number').shape[0] < max_reviews_in_highest_bin:
                    loot_state_corresponding_bin_number = dataframe_state.query('funniness_category==@bin_number')
                    # the number of reviews for a class that is being filled cannot exceed the number of reviews 
                    # of the class with the most reviews prior to filling
                    num = loot_state_corresponding_bin_number.shape[0] + df_basis.query('funniness_category==@bin_number').shape[0]
                    print(str(num) + " / " + str(max_reviews_in_highest_bin))
                    if loot_state_corresponding_bin_number.shape[0] + df_basis.query('funniness_category==@bin_number').shape[0] <= max_reviews_in_highest_bin:
                        df_basis = pd.concat([df_basis, loot_state_corresponding_bin_number], ignore_index=True)
                    else:
                        # only fill as many reviews as the class with the most reviews has
                        number_of_rows_to_add = max_reviews_in_highest_bin - df_basis.query('funniness_category==@bin_number').shape[0]
                        df_basis = pd.concat([df_basis, loot_state_corresponding_bin_number.head(number_of_rows_to_add)], ignore_index=True)
                # else:
                #     break
    
    
    # print stuff
    print("\nshape of illinois df after filling: ", df_basis.shape)
    print("bins: ", bins)
    print("after filling: ", Counter(df_basis['funniness_category'].tolist()))
    #return dict_dfs_loot_states


# execute function
loot_states = ["Wisconsin", "Ohio", "Nevada", "Quebec", "Pennsylvania", "Ontario", "Alberta", "Arizona", "North Carolina", "New York"]
fill_bins("Illinois", loot_states)
    
    