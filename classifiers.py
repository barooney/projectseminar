# -*- coding: utf-8 -*-

# third party modules
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier




def train_and_predict_multinomial_naive_bayes(features, labels):
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
    
    return labels_train, y_train_pred


def train_and_predict_softmax_logistic_regression(features, labels):
    
    
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
    
    # report
    print("\nReport:")
    print(classification_report(labels_train, y_train_predict))
      
    print("\nScores come here:----------\n")
    print("accuracy:")
    print(accuracy_score(labels_train, y_train_predict))
    print("precision:")
    print(precision_score(labels_train, y_train_predict, average="weighted"))  
    print("recall:")    
    print(recall_score(labels_train, y_train_predict, average="weighted"))   
    print("f1 score:")    
    print(f1_score(labels_train, y_train_predict, average="weighted"))  
    
    return labels_train, y_train_predict



def train_and_predict_random_forests(features, labels):
    
     # scicit learn shuffles data by default   
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=666)
    
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
   
    # perform cross validation on the trainig set
    y_train_predict = cross_val_predict(rnd_clf, features_train, labels_train, cv=3)
   
    # report
    print("\nReport:")
    print(classification_report(labels_train, y_train_predict))
      
    print("\nScores come here:----------\n")
    print("accuracy:")
    print(accuracy_score(labels_train, y_train_predict))
    print("precision:")
    print(precision_score(labels_train, y_train_predict, average="weighted"))  
    print("recall:")    
    print(recall_score(labels_train, y_train_predict, average="weighted"))   
    print("f1 score:")    
    print(f1_score(labels_train, y_train_predict, average="weighted"))  
    
    return labels_train, y_train_predict
   
   