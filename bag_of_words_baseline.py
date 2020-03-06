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



# Choose the state train a naive Bayes classifier for with bag of words absolutely frequency represention
parser = argparse.ArgumentParser(description="The script trains a naive Bayes classifier in order to predict one of several\
                                 mutually exclusive 'funniness' categories. Each text is represented as a text sparse text vector\
                                     counting ablsolute frequencies for each word in the text. All other words from the overall\
                                         vocabulary that do not occur in the text get a zero.\
                                 One parameter is required, the state: python3 ./bag_of_words_baseline.py <state>.")
parser.add_argument("state_input", help="Enter the state to separate data for: for example,\
                    python3 ./2-separate_labeled_and_unlabeled.py Illinois",
                    type=str)
parser.add_argument("--use_stop_words", help="use all reviews WITH stop words instead of the ones adjusted for stop words", action="store_true")
args = parser.parse_args()
STATE_TO_FILTER = args.state_input
df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json', lines=True)
if args.use_stop_words:
    print("Reviews WITH stop words are being used now.")   
else:
    print("Reviews WIHTOUT stop words are being used now.")


#df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews_zipf.json', lines=True)
min_funny = 0
max_funny = df['funny'].max()


# Sturge's Rule:
# K = 1 + 3. 322 log(N)
# where:
# K: num of bins
# N: num of observations

# calc sturges rule
# num_bins = 1 + 3.322 * np.log10(len(df))
# print(num_bins)


# Doane's formula
# k = 1 + log2(n) + log2(1 + ( |g_1| / sigma_g_1))
# g_1 = ( (6(n-2)) / ((n+1)(n+3)) )^0.5

# calc doane's formulara
N = len(df)
sigma_g_1 = math.sqrt(((6*(N-2)) / ((N+1)*(N+3))))
num_bins = 1 + math.log2(N) + math.log2(1+ abs(kurtosis(df['funny'].tolist())/sigma_g_1))
print(num_bins)


######
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
    
    # create funniness categories
    # 6 categories of funniness

    #   1  |  2  |  3  |  4   |   5   |  6
    #  -------------------------------------
    #   0  | 1-2 | 3-5 | 6-10 | 11-40 | +40
    
    # TODO: Good bin size?
    bins = [min_funny-1]
    samples = len(df)
    for i in range(1, int(num_bins)):
        bins.append(int(max_funny/num_bins*i))
    bins.append(max_funny+1)
    labels = range(0, len(bins)-1)
    print(bins)
    print(set(labels))
    #df['funniness_category'] = pd.cut(df.funny, bins=bins, labels=labels)
    #df['funniness_category'] = pd.cut(df.funny, bins=[-1,0,1,2,3,4,5,max_funny], labels=[1,2,3,4,5,6,7])
    
    
    df_shuffled = df.sample(frac=1)
    
    
    #labels = df_shuffled['funny'].values 
    labels = np.array(df_shuffled['funniness_category'].values)
    texts = df_shuffled['text'].values 
    
    # histogram 
    X = df['funny'].tolist()
    #n, bins, patches = plt.hist(X, int(num_bins), facecolor='blue', density=True)
    #n, bins, patches = plt.hist(X, [0,1,2,3,4,5,6,max_funny], facecolor='blue', density=True)
    n, bins, patches = plt.hist(X, [0,1,2,3,6,max_funny], edgecolor='white', density=True)
    patches[0].set_facecolor('b')   
    patches[1].set_facecolor('green')
    patches[2].set_facecolor('yellow')
    patches[3].set_facecolor('black') 
    patches[4].set_facecolor('r')
    
    plt.title('histogram')
    plt.xlabel('funny votes')
    plt.ylabel('frequency densitiy')
    #plt.show()
    plt.savefig('./doc/images/density_' + name + '.pdf', format='pdf')
    plt.close()
    
    #### test
    #sys.exit("nur bis hier")
    #####
    
    
    ####################################
    
    vectorizer = CountVectorizer(binary=False, stop_words='english')
    vectorizer.fit(texts)
    
    #print(vectorizer.vocabulary_.keys())  
    #print(vectorizer.get_feature_names())
    
    vectorizer.transform(texts)
      
    features = np.array(pd.DataFrame(vectorizer.transform(texts).toarray(), columns=sorted(vectorizer.vocabulary_.keys())))
      
    #df[df['text'].str.contains('黄鳝')]
    
    
    ################ train classifier #############
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=20)
    
    # 70% Trainingsdaten 
    print("features train:\n" ,features_train)#.reshape(-1,1))
    print("labels train:\n" ,labels_train)
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
    


    print("Confusion Matrix for cross validation on training set is saved now...\nTest set remained untouched.")
    
    # plot_confusion_matrix from mlxtend module is used in the following lines. 
    # and not scikits learn's function with the same name
    
    conf_mx = confusion_matrix(labels_train, y_train_pred)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
                                    cmap=plt.cm.Greys,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True)
    ax.set_title("Confusion matrix")

    print("Confusion Matrix for cross validation on training set.\nTest set remained untouched.")
    print(conf_mx)
    plt.title("Confusion Matrix " + ("(no condition)" if name == 'no_cond' else ""))
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    #plt.show()

    plt.savefig('./doc/images/confusion_matrix_' + name + '.pdf', format='pdf')
    plt.savefig('./doc/images/confusion_matrix_' + name + '.png', format='png')
    
    
    ## plotting confusion matrix with errors
    
    
    # # full the diagonal with zeros in order to keep only the errors:
    # mlxtend's plot_confusion_matrix has a built in function to show relative frequency, 
    # no need to divide each cell by the row sum
    np.fill_diagonal(conf_mx, 0)
    print(conf_mx)
   
    fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
                                    cmap=plt.cm.Greys,
                                  colorbar=True,
                                  show_absolute=False,
                                  show_normed=True)
    ax.set_title("Confusion matrix showing errors")
    print("Confusion Matrix showing only the errors is saved now..")
    #plt.figure(1)
    plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.pdf', format='pdf')
    plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.png', format='png')
    
    
    
    ###############################################################################################
    ############################################ old stuff ########################################
    
    # title = "Confusion matrix with abssolute frequencies"
    # class_names = [1,2,3,4,5]
    # disp = plot_confusion_matrix(classifier_fitted, labels_train.reshape(-1, 1), y_train_pred,
    #                              display_labels=class_names,
    #                              cmap=plt.cm.Blues,
    #                              normalize=None)
    # disp.ax_.set_title(title)
    # print(title)
    # print(disp.confusion_matrix)
    # plt.show()
    
    #conf_mx = confusion_matrix(labels_train, y_train_pred)
    # print(conf_mx)
    # plt.matshow(c, cmap=plt.cm.gray)
    # #plt.show()
    # plt.savefig('./doc/images/confusion_matrix_' + name + '.pdf', format='pdf')
    # plt.savefig('./doc/images/confusion_matrix_' + name + '.png', format='png')
     
    # # devide each cell of the confusion matrix by the total number of reviews corresponing to a class
    # # to get only relative numbers in case classes have an uneuqal number of reviews
    # # each row sum is equal to the total number of reviews in a class
    # row_sums = conf_mx.sum(axis=1, keepdims=True)
    # norm_conf_mx = conf_mx / row_sums
    # # full the diagonal with zeros in order to keep only the errors:
    # np.fill_diagonal(norm_conf_mx, 0)
    
    # print("Confusion Matrix showing only the errors:")
    # plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    # plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.pdf', format='pdf')
    # plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.png', format='png')
    
    
    
if __name__ == "__main__": 
    print("no condition:\n")
    train_model_baseline(df, 'no_cond')       # no condition, i.e. whole Illinois set, regardlass of all votes zero 
    print("-------------------------------\n")
    print("with condition:\n")
    train_model_baseline(labeled, 'labeled')   # with condition, see diagram
