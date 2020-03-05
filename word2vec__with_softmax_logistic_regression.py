# -*- coding: utf-8 -*-

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


# install English language first via: python -m spacy download en_core_web_sm



df = pd.read_json('./data/intermediate/' + 'random-small_reviews.json', lines=True)
#df = df.head(100)
print(df.shape)

nlp = spacy.load("en_core_web_sm")
# Create and register a new `tqdm` instance with `pandas`
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()
#print("\nUsing spacy's sophisticated tokenizer (compared to nltk). Unfortunately spacy performs a linguistic roundhouse kick, "
#  "meaning it also performs POS Tagging, dependency analysis and so on.")

#tokenized_texts = df['text'].progress_apply(lambda text_as_string: [token.text for token in nlp(text_as_string)])
tokenized_texts = df['text'].progress_apply(lambda text_as_string: nltk.word_tokenize(text_as_string))
print(tokenized_texts)


list_of_lists = tokenized_texts.tolist()

#print(list_of_lists)

# Das Modell trainieren

model = gensim.models.Word2Vec(list_of_lists, min_count=1,size=300,workers=3)

# Modell braucht tokenisierte Sätze/ eine Liste von Listen
# min_count = minimale Anzahl: Nur Wörter mit einer größeren Frequenz als dem angegebenen Wert werden berücksichtigt
# size = Größe des  Vectors (Anzahl der der Dimensionen)
# window = Fenstergröße (Max Distanz zwischen Target- und Kontextwort, Standard = 5) 
# workers = Anzahl der Trainingsdurchläufe

# Das gelernte Vokabular
words = model.wv.vocab

#dict_word_vector = {word:model["words"] for word in words.keys()}
#print(dict_word_vector)

# den Vektor eines Wortes bekommen?
cheese = model['cheese']
#dog
print("length word vector:", cheese.size)
print(type(cheese))

# Kosinus-Ähnlichkeit von zwei Wörtern?
#model.similarity("girl", "woman")
#print(model.similarity("food", "cheese"))

# Wenn Cos = 0: Vektoren sind sehr unähnlich
# Wenn Cos = 1: Vektoren sind identisch  

# Die ähnlichsten Wörter?
w1 = "beer"
#print(model.wv.most_similar(positive=w1))


## begin training the classifier


# get the mean vector of all word vectors per text
df["vectorized_texts"] = tokenized_texts.progress_apply(lambda tokens: np.array(np.mean([model[token] for token in tokens])))

# alternative is using tf-idf



print(df.head())

print(df["vectorized_texts"])



features = np.array(df["vectorized_texts"]).reshape(-1,1)
labels = df['funniness_category'].values

# scicit learn shuffles data by default
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=666)



# testaaray = np.array([1,2,4,67,3,-2,4,0,0,434,-1,2])
# testaaray2 = np.array([1,77,4,7,-3,-2,4,0,88,434,-0,2])
# print(np.mean([testaaray,testaaray2]))

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", verbose=0, C=10)
#softmax_reg.fit(features_train, labels_train)

# perform cross validation on the trainig set
y_train_predict = cross_val_predict(softmax_reg, features_train, labels_train, cv=3)

#new_f1_scorer = make_scorer(f1_score, average = 'weighted')


print("\nScores come here:----------\n")
print("accuracy:")
print(accuracy_score(labels_train, y_train_predict))
print("precision:")
print(precision_score(labels_train, y_train_predict, average="weighted"))  
print("recall:")    
print(recall_score(labels_train, y_train_predict, average="weighted"))   
print("f1 score:")    
print(f1_score(labels_train, y_train_predict, average="weighted"))  

                             
#labels_pred = softmax_reg.predict(features_validate)
     

# report
#print("\nReport:")
#print(classification_report(labels_validate, labels_pred))