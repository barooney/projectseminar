#modules

# standard library
import argparse
import sys
import math
import os 
import re

# third party modules
import pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


# application specifc
from create_small_test_sample import create_test_sample
from histogram import create_histograms
from confusion_matrices import create_confusion_matrices
from classifiers import train_and_predict_multinomial_naive_bayes



# Choose the state train a naive Bayes classifier for with bag of words absolutely frequency represention
parser = argparse.ArgumentParser(description="The script trains a naive Bayes classifier using bag of words as feature representation.\
                                 Bin sizes as a list can be given as an argument so that a random small sample based on all reviews (already \
                                      adjusted for stop words) is created. If none is given, it is assumed the random small sample already exits.")
parser.add_argument('bins_input', help="Enter a list of bins: lower values excluded, upper values included,\n\
                    e.g.: python3 ./create_small_test_sample.py [-1,1,2,5,10]. If none is given it is assumed the sample file already exists."
                    , nargs='?')
args = parser.parse_args()
input_bins = args.bins_input


# create a suitable sample in terms of the desired bin sizes if bins are given

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



#print(type(input_bins))


# read random sample that is used for training 
STATE_TO_FILTER = [f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)][0]

df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER , lines=True)

min_funny = 0
max_funny = df['funny'].max()



###### with condition ## reviews that havent been rated at all are excluded
labeled = df.query('funny>0 or cool>0 or useful>0')
unlabeled = df.query('funny==0 and cool==0 and useful==0')
print(df.shape)
print(labeled.shape)
print(unlabeled.shape)

#labeled.to_json('./data/intermediate/Illinois_labeled.json', orient='records', lines=True)
#unlabeled.to_json('./data/intermediate/Illinois_unlabeled.json', orient='records', lines=True)


# #print(labeled.info())
#print(df.sort_values(by=['funny'], ascending=False))




##########################

def train_model_baseline(df, name):
    '''use bag of words as feature representation and naive Bayes as classifier'''

     
        
    #labels = df_shuffled['funny'].values 
    labels = np.array(df['funniness_category'].values)
    texts = df_shuffled['text'].values 
    
    # create histograms
    create_histograms(df, input_bins, name)
    
    
    
    #################################### get bag of words with a counting vector ####
    
    #vectorizer = CountVectorizer(binary=False, stop_words='english')
    vectorizer = CountVectorizer(binary=False)  
    
    features = vectorizer.fit_transform(texts)
      
    #df[df['text'].str.contains('黄鳝')]
    
    
    ##################################
    ### tf idf vectorizer ###### 
    
    # tfidf_vectorizer = TfidfVectorizer()
    # features = tfidf_vectorizer.fit_transform(texts)
    
    ##################
    
    
    ################ train classifier and predict with cross validation on training set #############

    labels_train, y_train_pred = train_and_predict_multinomial_naive_bayes(features, labels)
    

    ####### confusion matrix##
    if name == 'no_cond':
        create_confusion_matrices(labels_train, y_train_pred)
    else:
        create_confusion_matrices(labels_train, y_train_pred, condition=True)
    
    
    
    
if __name__ == "__main__": 
    print("no condition:\n")
    train_model_baseline(df, 'no_cond')       # no condition, i.e. whole Illinois set, regardlass of all votes zero 
    print("-------------------------------\n")
    print("with condition:\n")
    train_model_baseline(labeled, 'labeled')   # with condition, see diagram
