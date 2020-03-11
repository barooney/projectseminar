import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np


def create_confusion_matrices(labels_train, y_train_pred, name):
    print("Confusion Matrix for cross validation on training set is saved now...\nTest set remained untouched.")
    
    # plot_confusion_matrix from mlxtend module is used in the following lines. 
    # and not scikits learn's function with the same name
    
    conf_mx = confusion_matrix(labels_train, y_train_pred)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
                                    cmap=plt.cm.Blues,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True)
    ax.set_title("Confusion matrix " + "(no condition)" if name == 'no_cond' else "Confusion matrix " + "(labeled)")

    print("Confusion Matrix for cross validation on training set.\nTest set remained untouched.")
    print(conf_mx)

    plt.savefig('./doc/images/confusion_matrix_' + name + '.pdf', format='pdf')
    plt.savefig('./doc/images/confusion_matrix_' + name + '.png', format='png')
    
    
    ## plotting confusion matrix with errors
    
    
    # # full the diagonal with zeros in order to keep only the errors:
    # mlxtend's plot_confusion_matrix has a built in function to show relative frequency, 
    # no need to divide each cell by the row sum
    np.fill_diagonal(conf_mx, 0)
    print(conf_mx)
   
    fig, ax = plot_confusion_matrix(conf_mat=conf_mx,
                                    cmap=plt.cm.Blues,
                                  colorbar=True,
                                  show_absolute=False,
                                  show_normed=True)
    ax.set_title("Confusion matrix showing errors " + "(no condition)" if name == 'no_cond' else "Confusion matrix showing errors " + "(labeled)")
    print("Confusion Matrix showing only the errors is saved now..")
    #plt.figure(1)
    plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.pdf', format='pdf')
    plt.savefig('./doc/images/confusion_matrix_errors_' + name + '.png', format='png')