import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np


def create_confusion_matrices(labels_train, y_train_pred, feature_representation="bag of words", classifier_type="Naive Bayes", condition=False):
    print("Confusion Matrix for cross validation on training set is saved now...\nTest set remained untouched.")
    
   
     # get confusion matrix
    conf_mx = confusion_matrix(labels_train, y_train_pred)
    
    
    def save_confusion_matrix(conf_mx, errors=False):
        
         # plot_confusion_matrix from mlxtend module is used in the following lines. 
         # and not scikits learn's function with the same name          
        conf_mx = confusion_matrix(labels_train, y_train_pred)
        
        
        # # full the diagonal with zeros in order to keep only the errors:
        # mlxtend's plot_confusion_matrix has a built in function to show relative frequency, 
        # no need to divide each cell by the row sum
        if errors != False:
             np.fill_diagonal(conf_mx, 0)
        
        fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
                                        cmap=plt.cm.Blues,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=True)
        
        cond = "(no condition)" if condition == False else "(labeled)"
        matrix_name = "Confusion matrix " if errors == False else "Confusion matrix errors only "
        title_for_filename = matrix_name + feature_representation + " " + classifier_type + " " + cond
        if classifier_type =="Naive Bayes":
            title_name = matrix_name + "Baseline "  + cond
        else:
            title_name = matrix_name + " " + cond
        ax.set_title(title_name)
    
        print("Confusion Matrix for cross validation on training set.\nTest set remained untouched.")
        print(conf_mx)
    
        plt.savefig('./doc/images/' + title_for_filename.lower().replace(" ", "_") + '.pdf', format='pdf')
        plt.savefig('./doc/images/' + title_for_filename.lower().replace(" ", "_") + '.png', format='png')
    
    
    
    # save normal confusion matrix and the one showing errors only
    save_confusion_matrix(conf_mx)
    save_confusion_matrix(conf_mx, errors=True)
    
    
    
    
    
    
    