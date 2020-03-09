#modules

# standard library
import argparse
import sys
import math

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
#from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# application specifc
from create_small_test_sample import create_test_sample
from histogram import create_histograms
from confusion_matrices import create_confusion_matrices



# Choose the state train a naive Bayes classifier for with bag of words absolutely frequency represention
parser = argparse.ArgumentParser(description="The script trains a naive Bayes classifier using bag of words as feature representation.\
                                 Bin sizes as a list can be given as an argument so that a random small sample based on all reviews (already \
                                      adjusted for stop words) is created. If none is given, the default bin size is used.")
parser.add_argument('bins_input', help="Enter a list of bins: lower values excluded, upper values included,\n\
                    e.g.: python3 ./create_small_test_sample.py [-1,1,2,5,10]. Default bins are: [-1,1,3,5,10,100,9999999999999999]"
                    , nargs='?', default=[-1,1,3,5,10,100,9999999999999999], type=str)
args = parser.parse_args()
input_bins = args.bins_input


STATE_TO_FILTER  = "random-small"
df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews_zipf.json', lines=True)
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
    
   # create a suitable sample in terms of the desired bin sizes
    create_test_sample(input_bins)
      
    df_shuffled = df.sample(frac=1)
    
    
    #labels = df_shuffled['funny'].values 
    labels = np.array(df_shuffled['funniness_category'].values)
    texts = df_shuffled['text'].values 
    
    # create histograms
    create_histograms(df, input_bins, name)
    
    
    
    #################################### get bag of words with a counting vector ####
    
    vectorizer = CountVectorizer(binary=False, stop_words='english')
    vectorizer.fit(texts)
    
    #print(vectorizer.vocabulary_.keys())  
    #print(vectorizer.get_feature_names())
    
    vectorizer.transform(texts)
      
    features = np.array(pd.DataFrame(vectorizer.transform(texts).toarray(), columns=sorted(vectorizer.vocabulary_.keys())))
      
    #df[df['text'].str.contains('黄鳝')]
    
    
    ################ train classifier #############
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=20)
    
    # 80% Trainingsdaten 
    print("features train:\n" ,features_train)#.reshape(-1,1))
    print("labels train:\n" ,labels_train)
    # #
    # ## 20 % Testdaten
    # print(features_test)
    # print(labels_test)
    
     
    # # create classifier: 
    # GaussianNB did not produce a result, hence MultinomialNB was used
    gnb = MultinomialNB()
    
    # # train classifier with training set: 
    gnb.fit(features_train, labels_train)
    
    # # use classifier on test set: 
    labels_pred = gnb.predict(features_test)
    print("Result for test set:")
    print(labels_pred)
    
    # # How well is the classifier performing? 
    print("\nAccuracy:")
    print(accuracy_score(labels_test, labels_pred))
    #print(gnb.score(features_test, labels_test))
    
    # report
    print("\nReport:")
    print(classification_report(labels_test, labels_pred))

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform()

    y_train_pred = cross_val_predict(gnb, features_train, labels_train, cv=3)
    

    ####### confusion matrix##
    create_confusion_matrices(labels_train, y_train_pred, name)
    
    
    
    
if __name__ == "__main__": 
    print("no condition:\n")
    train_model_baseline(df, 'no_cond')       # no condition, i.e. whole Illinois set, regardlass of all votes zero 
    print("-------------------------------\n")
    print("with condition:\n")
    train_model_baseline(labeled, 'labeled')   # with condition, see diagram
