import collections
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import spacy
import nltk
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier




def train_doc2vec(df):

    annotations = []
    for ent in tqdm(df['text']):
    	#spacy
    	# doc = nlp(ent)
    	#nltk
    	doc = nltk.word_tokenize(ent)
    
    	annotations.append(doc)
    df['annotations'] = annotations
    
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['annotations'])]
    num_docs = len(documents)
    train_corpus = documents[:(int(num_docs * 0.8))]
    test_corpus = [doc[0] for doc in documents[(int(num_docs * 0.8)):]]
    print(train_corpus[:2])
    print(test_corpus[:2])
    print("num docs: " + str(num_docs))
    print("num train docs: " + str(len(train_corpus)))
    print("num test docs: " + str(len(test_corpus)))
    model = Doc2Vec(vector_size=512, window=2, min_count=1, workers=4)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    
    # bis hier hin verstanden, rest aus dem Tutorial kopiert:
    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
    
    # ranks = []
    # second_ranks = []
    # for doc_id in range(len(train_corpus)):
    #     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    #     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #     rank = [docid for docid, sim in sims].index(doc_id)
    #     ranks.append(rank)
    
    #     second_ranks.append(sims[1])
    
    # #print("sims :\n" , sims)
    
    
    
    # counter = collections.Counter(ranks)
    # print(counter)
    
    # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
        

############################################ training classifier to get ducement vectors ###############################
    
    df["docvec"] = df["annotations"].apply(lambda tokens: model.infer_vector(tokens))

    #print("hieeer######################")
    #print(df["docvec"])

    return model, df



######################## zu ende ab hier: der code ist umgezogen ########################
######################################################

# features = df["docvec"].tolist()
# labels = df['funniness_category'].values

# # scicit learn shuffles data by default
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=666)



# # testaaray = np.array([1,2,4,67,3,-2,4,0,0,434,-1,2])
# # testaaray2 = np.array([1,77,4,7,-3,-2,4,0,88,434,-0,2])
# # print(np.mean([testaaray,testaaray2]))

# ####
# ## softmax logistic regression

# softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", verbose=0, C=10)
# #softmax_reg.fit(features_train, labels_train)

# # perform cross validation on the trainig set
# y_train_predict = cross_val_predict(softmax_reg, features_train, labels_train, cv=3)

# #new_f1_scorer = make_scorer(f1_score, average = 'weighted')


# print("\nScores come here:----------\n")
# print("accuracy:")
# print(accuracy_score(labels_train, y_train_predict))
# print("precision:")
# print(precision_score(labels_train, y_train_predict, average="weighted"))  
# print("recall:")    
# print(recall_score(labels_train, y_train_predict, average="weighted"))   
# print("f1 score:")    
# print(f1_score(labels_train, y_train_predict, average="weighted"))  

  
# ############ confusion matrix #########

# print("Confusion Matrix for cross validation on training set is saved now...\nTest set remained untouched.")
    
# # plot_confusion_matrix from mlxtend module is used in the following lines. 
# # and not scikits learn's function with the same name
    
# conf_mx = confusion_matrix(labels_train, y_train_predict)
# fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
#                                     cmap=plt.cm.Blues,
#                                 colorbar=True,
#                                 show_absolute=True,
#                                 show_normed=True)
# ax.set_title("Confusion matrix")
# #plt.show()
# print("Confusion Matrix for cross validation on training set.\nTest set remained untouched.")
# print(conf_mx)

# plt.savefig('./doc/images/doc2vec-confusion.pdf', format='pdf')
# plt.savefig('./doc/images/doc2vec-confusion.png', format='png')
    
    
# ## plotting confusion matrix with errors
    
    
# # # full the diagonal with zeros in order to keep only the errors:
# # mlxtend's plot_confusion_matrix has a built in function to show relative frequency, 
# # no need to divide each cell by the row sum
# np.fill_diagonal(conf_mx, 0)
# print(conf_mx)
   
# fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
#                                     cmap=plt.cm.Blues,
#                                   colorbar=True,
#                                   show_absolute=False,
#                                   show_normed=True)
# ax.set_title("Confusion matrix showing errors")
# print("Confusion Matrix showing only the errors is saved now..")
# #plt.show()
# plt.savefig('./doc/images/doc2vec-confusion-errors.pdf', format='pdf')
# plt.savefig('./doc/images/doc2vec-confusion-errors.png', format='png')

                           
# #labels_pred = softmax_reg.predict(features_validate)
     
# # report
# #print("\nReport:")
# #print(classification_report(labels_validate, labels_pred))



# ##### random forests ##################

# rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
# rnd_clf.fit(features_train,labels_train)

# y_pred_rf = rnd_clf.predict(features_test)

# # report
# # print("\nReport:")
# # print(classification_report(features_test, y_pred_rf))

# conf_mx = confusion_matrix(labels_test, y_pred_rf)
# fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
#                                     cmap=plt.cm.Blues,
#                                 colorbar=True,
#                                 show_absolute=True,
#                                 show_normed=True)
# ax.set_title("Confusion matrix")
# plt.savefig('doc/images/doc2vec-confusion-random-forest.png', format='png')
# plt.savefig('doc/images/doc2vec-confusion-random-forest.pdf', format='pdf')
# print("Confusion Matrix for cross validation on training set.\nTest set remained untouched.")
# print(conf_mx)


# print("\nScores come here:----------\n")
# print("accuracy:")
# print(accuracy_score(labels_test, y_pred_rf))
# print("precision:")
# print(precision_score(labels_test, y_pred_rf, average="weighted"))  
# print("recall:")    
# print(recall_score(labels_test, y_pred_rf, average="weighted"))   
# print("f1 score:")    
# print(f1_score(labels_test, y_pred_rf, average="weighted")) 

# model.save('data/intermediate/doc2vec.model')