#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:58:05 2019

@author: painkiller
"""

# imports
from collections import Counter
#import reverse_geocoder
import itertools 
import json
import nltk
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# DEFAULTS

# define folder structure
base_path = os.getcwd()
data_path = base_path + '/data'
intermediate_data_path = data_path + '/intermediate'

try:
    os.mkdir(data_path)
    print("Folder created.")
except FileExistsError:
    print("Folder already exists.")

try:
    os.mkdir(intermediate_data_path)
    print("Folder created.")
except FileExistsError:
    print("Folder already exists.")
    
    
# DATA MODELS

# Business
class Business:
    def __init__(self, json):
        self.__dict__ = json

businesses = dict()

# Review
class Review:
    def __init__(self, json):
        self.__dict__ = json

reviews = dict()



#########################################################################
# open reviews from Illinois

with open(intermediate_data_path + '/Illinois_reviews.json', encoding="utf8") as reviews_file:
    for l in tqdm(reviews_file.readlines()):
        r = Review(json.loads(l))
        reviews[r.review_id] = r

        
def compute_zipf_table(WORDS, sort_parameters=("rank", "ascending"), num_rows=10):
    ' WORDS = list of words'
    zipf_values = [(wort, frequ, rank, frequ*rank) for rank, (wort, frequ) in enumerate(Counter(WORDS).most_common(len(WORDS)), 1)]
    return zipf_values


def get_words(review_dict):
    #return [word for review_obj in review_dict.values() for word in nltk.word_tokenize(review_obj.text)]
    all_words = []
    for review_obj in tqdm(review_dict.values()):
        for word in nltk.word_tokenize(review_obj.text):
            all_words.append(word)
    return all_words




WORDS_zipf = compute_zipf_table(get_words(reviews))



all_words = get_words(reviews)


frequencies = [ele[1] for ele in WORDS_zipf]    
ranks = [ele[2] for ele in WORDS_zipf]   


plt.loglog(ranks, frequencies)
plt.xlabel('Ranks')
plt.ylabel('Frequencies')
plt.grid()
#plt.xticks(indexes + 0.5, plotting_counting.keys(), rotation='vertical')
plt.show()

type_token_ratio = len(WORDS_zipf)/len(all_words) # no stems, each unique orthographic word is a type
print("Type-token: ", type_token_ratio)

# remove the 50 most frequent words except for good, food, place, cos they are relevant for the review
words_without_stop_words = [word[0] for word in WORDS_zipf if word[2]>50 if word not in ('good', 'food', 'place') ]
print(words_without_stop_words[:100])

