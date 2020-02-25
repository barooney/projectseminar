#!/usr/bin/python3

import pandas as pd
import sys
import argparse
import os

from models import Business, Review



# Choose the state to separate labels for words for
parser = argparse.ArgumentParser(description="The script separates the reviews of a given state into labeld and unlabeled data.\
                                 One parameter is required, the state: python3 ./3-processing_stop_words.py <state>. \
                                     It outputs labeled and unlabeled data for both reviews with and without stop words.\
                                         Make sure that step 2 '2-preprocessing_stop_words' has been carried out before.")
parser.add_argument("state_input", help="Enter the state to separate data for: for example,\
                    python3 ./2-separate_labeled_and_unlabeled.py Illinois",
                    type=str)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input



def separate_labeled_unlabeled(df, zipfed=False):
    print(STATE_TO_FILTER)   
    labeled = df.query('funny>0 or cool>0 or useful>0')
    unlabeled = df.query('funny==0 and cool==0 and useful==0')
    print(df.shape)
    print(labeled.shape)
    print(unlabeled.shape)
    name_str = ""
    if zipfed == True:
        name_str = "_zipf"
    labeled.to_json('./data/intermediate/' + STATE_TO_FILTER + name_str + '_labeled.json', orient='records', lines=True)
    unlabeled.to_json('./data/intermediate/' + STATE_TO_FILTER + name_str + '_unlabeled.json', orient='records', lines=True)



# read ORIGINAL review file
df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json', lines=True)
separate_labeled_unlabeled(df)

if not os.path.exists('./data/intermediate/' + STATE_TO_FILTER + '_reviews_zipf.json'):
    raise AssertionError("Unable to separate labeled and unlabled review data for the review file that was adjusted for stop words.")
else:
    df_2 = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews_zipf.json', lines=True)
    separate_labeled_unlabeled(df_2, zipfed=True)
    

    




