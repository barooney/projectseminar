# -*- coding: utf-8 -*-

from tqdm import tqdm

import gensim
from gensim.models import Word2Vec
import spacy
import nltk
import numpy as np
import pandas as pd


def train_word2vec(df):
    '''
    train word vectors with the Word2Vec package from gensim library: df must a pandas DataFrame
    returns model with the learned word vectors and a datafram which now has a new column with a mean word vector for each text
    '''
    # Create and register a new `tqdm` instance with `pandas`
    # (can use tqdm_gui, optional kwargs, etc.)
    tqdm.pandas()

    try:
        # Modell laden?
        model = gensim.models.Word2Vec.load('./data/models/word2vec_model.bin')
    except:
         
        ## load all reviews to train word vectors with
        df_all_reviews = pd.read_json('./data/intermediate/zipf_all_reviews.json', lines=True)
        
        nlp = spacy.load("en_core_web_sm")
        
        #print("\nUsing spacy's sophisticated tokenizer (compared to nltk). Unfortunately spacy performs a linguistic roundhouse kick, "
        #  "meaning it also performs POS Tagging, dependency analysis and so on.")
        
        #tokenized_texts = df['text'].progress_apply(lambda text_as_string: [token.text for token in nlp(text_as_string)])
        tokenized_texts = df_all_reviews['text'].progress_apply(lambda text_as_string: nltk.word_tokenize(text_as_string))
        #print(tokenized_texts)
            
        list_of_lists = tokenized_texts.tolist()
        
        
        # Das Modell trainieren
        
        model = gensim.models.Word2Vec(list_of_lists, min_count=1,size=300,workers=24)
        
        # Modell braucht tokenisierte Sätze/ eine Liste von Listen
        # min_count = minimale Anzahl: Nur Wörter mit einer größeren Frequenz als dem angegebenen Wert werden berücksichtigt
        # size = Größe des  Vectors (Anzahl der der Dimensionen)
        # window = Fenstergröße (Max Distanz zwischen Target- und Kontextwort, Standard = 5) 
        # workers = Anzahl der Trainingsdurchläufe
        
        model.save('./data/models/word2vec_model.bin')
        
        # Das gelernte Vokabular
        words = model.wv.vocab
        
        #dict_word_vector = {word:model["words"] for word in words.keys()}
        #print(dict_word_vector)
        
        # den Vektor eines Wortes bekommen?
        cheese = model['cheese']
        #dog
        #print("length word vector:", cheese.size)
        #print(type(cheese))
        
        # Kosinus-Ähnlichkeit von zwei Wörtern?
        #model.similarity("girl", "woman")
        #print(model.similarity("food", "cheese"))
        
        # Wenn Cos = 0: Vektoren sind sehr unähnlich
        # Wenn Cos = 1: Vektoren sind identisch  
        
        # Die ähnlichsten Wörter?
        w1 = "beer"
        #print(model.wv.most_similar(positive=w1))
        
        # get the mean vector of all word vectors per text
    df["vectorized_texts"] = df["text"].progress_apply(lambda tokens: np.array(np.mean([model[token] for token in nltk.word_tokenize(tokens)])))

    # alternative is using tf-idf
    
    return model, df
