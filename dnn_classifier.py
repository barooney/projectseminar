# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os, re, sys


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# application specifc
from create_small_test_sample import create_test_sample
from histogram import create_histograms
from confusion_matrices import create_confusion_matrices
from doc2vec import train_doc2vec



print(tf.__version__)



# Choose the state train a softmax logistic regression classifier
parser = argparse.ArgumentParser(description="The script trains a dnn classifier using doc2vec as feature representation.\
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



#### train classifier and stuff ####

dnn_classifier = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6)
])

dnn_classifier.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


 # get feature representation via doc2vec
model, df = train_doc2vec(df)
    
features = df["docvec"].tolist()
labels = df['funniness_category'].values

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=20)
    
features_train = tf.convert_to_tensor(
    features_train, dtype=None, dtype_hint=None, name=None
)

labels_train = tf.convert_to_tensor(
    labels_train, dtype=None, dtype_hint=None, name=None
)

features_test = tf.convert_to_tensor(
    features_test, dtype=None, dtype_hint=None, name=None
)


labels_test = tf.convert_to_tensor(
    labels_test, dtype=None, dtype_hint=None, name=None
)

# 80% Trainingsdaten 
print("features train:\n" ,features_train)#.reshape(-1,1))
print("labels train:\n" ,labels_train)


dnn_classifier.fit(features_train, labels_train, epochs=10)


test_loss, test_acc = dnn_classifier.evaluate(features_test,  labels_test, verbose=2)

print('\nTest accuracy:', test_acc)
