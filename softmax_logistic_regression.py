# -*- coding: utf-8 -*-


# standard library
import os
import sys
import argparse
import re

# third party
import spacy
import pandas as pd
import gensim
from gensim.models import Word2Vec
import nltk
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# application specifc
from word2vec import train_word2vec
from create_small_test_sample import create_test_sample
from histogram import create_histograms
from confusion_matrices import create_confusion_matrices
from classifiers import train_and_predict_softmax_logistic_regression


# Choose the state train a softmax logistic regression classifier
parser = argparse.ArgumentParser(description="The script trains a softmax logistic regression classifier using word2vec as feature representation.\
                                 Bin sizes as a list can be given as an argument so that a random small sample based on all reviews (already \
                                      adjusted for stop words) is created. If none is given, it is assumed the random small sample already exits.")
parser.add_argument('bins_input', help="Enter a list of bins: lower values excluded, upper values included,\n\
                    e.g.: python3 ./create_small_test_sample.py [-1,1,2,5,10]. If none is given it is assumed the sample file already exists."
                    , nargs='?')
args = parser.parse_args()
input_bins = args.bins_input


# create a suitable sample in terms of the desired bin sizes if bins are given or use an existing sample instead
if len(sys.argv) > 1:
    print("a new sample is created using bins: {}".format(input_bins))
    input_bins = eval(input_bins)
    create_test_sample(input_bins)
else:
    print("a sample file that already exists is going to be used.")
    assert len([f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)]), "Either there is no file having the bin sizes in its name or there are multiple such files."
    file_name = [f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)][0]
    input_bins = [int(number) for number in file_name.split("_")[1].split(".")]
    print(input_bins)



# read random sample that is used for training 
STATE_TO_FILTER = [f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)][0]

df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER , lines=True)

###### with condition ## reviews that haven't been rated at all are excluded
labeled = df.query('funny>0 or cool>0 or useful>0')
unlabeled = df.query('funny==0 and cool==0 and useful==0')
print(df.shape)
print(labeled.shape)
print(unlabeled.shape)


def train_modelsoftmax_regression(df, name):

    # train word vectors using all reviews as data
    model, df  = train_word2vec(df)
    
    ### begin training the softmax logistic regression classifier and predict using cross validation on training set
    
    #print(df.head())
    
    #print(df["vectorized_texts"])
    
    
    features = np.array(df["vectorized_texts"]).reshape(-1,1)
    labels = df['funniness_category'].values
    
    
    ###### train and predict with softmax logistic regression
    labels_train, y_train_predict = train_and_predict_softmax_logistic_regression(features, labels)
    
      
    ############ confusion matrix #########
    
    create_confusion_matrices(labels_train, y_train_predict, feature_representation="word2vec", classifier_type="Softmax regression", condition=False if name =='no_cond' else True)
    

###################

if __name__ == "__main__": 
    print("no condition:\n")
    train_modelsoftmax_regression(df, 'no_cond')       # no condition, i.e. whole Illinois set, regardlass of all votes zero 
    print("-------------------------------\n")
    print("with condition:\n")
    train_modelsoftmax_regression(labeled, 'labeled')   # with condition, see diagram