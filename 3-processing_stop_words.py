#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:58:05 2019

@author: painkiller
"""

# imports

# standard library
from collections import Counter
import itertools 
import json
import os
import argparse
import sys

# third party modules
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt

# application specific modules
from models import Business, Review

# DEFAULTS


# Choose the state to filter stop words for
parser = argparse.ArgumentParser(description="The script filters stop words for a given state.\
                                 One parameter is required, the state: python3 ./3-processing_stop_words.py <state>. \
                                     For example, python3 ./3-processing_stop_words.py Illinois")
parser.add_argument("state_input", help="Enter the state to filter stop words for: for example,\
                    python3 ./3-processing_stop_words.py Illinois",
                    type=str)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input

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
    

businesses = dict()
reviews = dict()



#########################################################################
# open reviews from Illinois

with open(intermediate_data_path + '/' + STATE_TO_FILTER + '_reviews.json', encoding="utf8") as reviews_file:
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
# plt.show()

type_token_ratio = len(WORDS_zipf)/len(all_words) # no stems, each unique orthographic word is a type
print("Type-token: ", type_token_ratio)

# remove the 50 most frequent words except for good, food, place, cos they are relevant for the review
words_without_stop_words = [word[0] for word in WORDS_zipf if word[2]>50 and word[1] > 1 or word[0] in ('good', 'food', 'place') ]
stop_words = [zipf_tuple[0] for zipf_tuple in WORDS_zipf if zipf_tuple[2]<51 if zipf_tuple[0] not in ('good', 'food', 'place')]
print("\n")
print("no stoppys;", words_without_stop_words[:100])
print("\n")
print("stoppys:",stop_words)


#ys.exit("das wars erst mal nur bis zipf")

with open('./data/intermediate/' + STATE_TO_FILTER + '_zipf.json', 'w') as zipf_file:
    before = 0
    after = 0
    for r in tqdm(reviews):
        review_words = nltk.word_tokenize(reviews[r].text)   # we should also use the same tokenizing method here
        before += len(review_words)
        #print(r + " len before: " + str(len(review_words)))
        #intersect = list(set(review_words).intersection(words_without_stop_words))
        intersect = [word for word in review_words if word not in stop_words]
        after += len(intersect)
        #print(r + " len after: " + str(len(intersect)))
        reviews[r].text = " ".join(intersect)
        json.dump(reviews[r].__dict__, zipf_file)
        zipf_file.write("\n")
    print("Before applying Zipf's law, there were " + str(before) + " words in the corpus.")
    print("After applying Zipf's law, there were " + str(after) + " words in the corpus.")