
# third party modules
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

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

df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews_zipf.json', lines=True)
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

def train_model_baseline(df):
    '''use bag of words as feature representation and naive Bayes as classifier'''
    
    # create funniness categories
    # 6 categories of funniness

    #   1  |  2  |  3  |  4   |   5   |  6
    #  -------------------------------------
    #   0  | 1-2 | 3-5 | 6-10 | 11-40 | +40
    
    # TODO: Good bin size?
    df['funniness_category'] = pd.cut(df.funny, bins=[-1,0,2,5,10,20,30,40,100], labels=[1,2,3,4,5,6,7,8])
    
    
    df_shuffled = df.sample(frac=1)
    
    
    #labels = df_shuffled['funny'].values 
    labels = df_shuffled['funniness_category'].values 
    texts = df_shuffled['text'].values 
    
   
    ####################################
    
    vectorizer = CountVectorizer(binary=False, stop_words='english')
    vectorizer.fit(texts)
    
    #print(vectorizer.vocabulary_.keys())  
    #print(vectorizer.get_feature_names())
    
    vectorizer.transform(texts)
      
    features = pd.DataFrame(vectorizer.transform(texts).toarray(), columns=sorted(vectorizer.vocabulary_.keys()))
      
    #df[df['text'].str.contains('黄鳝')]
    
    
    ################ train classifier #############
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=20)
    
    # 70% Trainingsdaten 
    # print(features_train)
    # print(labels_train)
    # #
    # ## 30 % Testdaten
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
    conf_mx = confusion_matrix(labels_train, y_train_pred)

    print("Confusion Matrix")
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
if __name__ == "__main__": 
    print("no condition:\n")
    train_model_baseline(df)       # no condition, i.e. whole Illinois set, regardlass of all votes zero 
    print("-------------------------------\n")
    print("with condition:\n")
    train_model_baseline(labeled)   # with condition, see diagram
