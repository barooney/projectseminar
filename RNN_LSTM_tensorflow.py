## Multi Class Text Classification with LSTM using TensorFlow 2.0
## Recurrent Neural Networks, Long Short Term Memory
## https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35


import argparse
import sys
import re
import os
import pandas as pd

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# application specifc
from create_small_test_sample import create_test_sample
from histogram import create_histograms
from confusion_matrices import create_confusion_matrices


# wenn ich die Datei in der Console ausführe, gibt's ne Fehlermeldung
# aber in Spyder (Editor mit Python IDE, der bei Anaconda dabei ist) geht's

# # Choose the state train a softmax logistic regression classifier
# parser = argparse.ArgumentParser(description="The script trains a dnn classifier using doc2vec as feature representation.\
#                                  Bin sizes as a list can be given as an argument so that a random small sample based on all reviews (already \
#                                       adjusted for stop words) is created. If none is given, it is assumed the random small sample already exits.")
# parser.add_argument('bins_input', help="Enter a list of bins: lower values excluded, upper values included,\n\
#                     e.g.: python3 ./create_small_test_sample.py [-1,1,2,5,10]. If none is given it is assumed the sample file already exists."
#                     , nargs='?')
# args = parser.parse_args()
# input_bins = args.bins_input


# # create a suitable sample in terms of the desired bin sizes if bins are given or use an existing sample instead
# if len(sys.argv) > 1:
#     print("a new sample is created using bins: {}".format(input_bins))
#     input_bins = eval(input_bins)
#     create_test_sample(input_bins)
# else:
#     print("a sample file that already exists is going to be used.")
#     assert len([f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)]), "Either there is no file having the bin sizes in its name or there are multiple such files."
#     file_name = [f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)][0]
#     input_bins = [int(number) for number in file_name.split("_")[1].split(".")]
#     print(input_bins)



# # read random sample that is used for training 
# STATE_TO_FILTER = [f for f in os.listdir('./data/intermediate/') if re.match(r'random.+_.+_reviews.json', f)][0]

# df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER , lines=True)

df = pd.read_json('./data/intermediate/' + "random-small_-1.0.1.2.5.10.999999_reviews.json" , lines=True)


###### with condition ## reviews that haven't been rated at all are excluded
labeled = df.query('funny>0 or cool>0 or useful>0')
unlabeled = df.query('funny==0 and cool==0 and useful==0')
print(df.shape)
print(labeled.shape)
print(unlabeled.shape)



#### train classifier and stuff ####

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


features = df["text"].tolist()
labels = np.array(df['funniness_category'].values)
    
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=20)

print(labels_train[1])   # a list of strings: each string is the text, i.e. the review


# get the indexes of all words in the vocab
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(features_train)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))


train_sequences = tokenizer.texts_to_sequences(features_train)
print(train_sequences[10])


# make all sequences have the same length
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

print(train_padded[10])


validation_sequences = tokenizer.texts_to_sequences(features_test)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)



model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(6, activation='softmax')
])
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(train_padded, labels_train, epochs=num_epochs, validation_data=(validation_padded, labels_test), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")