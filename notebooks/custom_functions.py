# Calculate metric

from sklearn.metrics import confusion_matrix
# for display dataframe
from IPython.display import display 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

#sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

# make scorer 
def get_kappa():
    kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
    return kappa_scorer


def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """Calculate and print out RMSE and R2 for train and test data
    Args:
        y_train (array): true values of y_train
        y_pred_train (array): predicted values of model for y_train
        y_test (array): true values of y_test
        y_pred_test (array): predicted values of model for y_test
    """

# write function displaying our model metrics
def our_metrics(y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, cmap='PuOr', annot=True);
        print('Model Metrics and Normalized Confusion Matrix:')
        print("_____________________")
    else:
        print('Model Metrics and Confusion Matrix without Normalization:')
        sns.heatmap(cm, cmap='PuOr', annot=True);
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("_____________________")
    print('Weighted Quadratic Kappa:', round(cohen_kappa_score(y_true, y_pred, weights='quadratic'), 4)) 
    
    
    # #print('F1-score:', round(f1_score(y_true, y_pred), 4))
    # print("_____________________")
    # print('Fbeta_score with beta=1.5:', round(fbeta_score(y_true, y_pred, beta=1.5), 4)) 
    # print("_____________________")
    # print('Fbeta_score with beta=2:', round(fbeta_score(y_true, y_pred, beta=2), 4)) 
    # print("_____________________")
    # print('Fbeta_score with beta=3:', round(fbeta_score(y_true, y_pred, beta=3), 4)) 
    # print("_____________________")
    # print('Recall', round(recall_score(y_true, y_pred), 4))
    # print("_____________________")
    # print('Specificity', round(recall_score(y_true, y_pred, pos_label=0), 4))


# def class_metrics_var_threshold(fitted_model, X, y_true, threshold=0.5, normalize=True):
#     """
#     function to compute confusion matrix and classification metrics based on passed threshold.
#     required arguments: fitted model, X_test, y_test, threshold (float, default >= 0.5)
#     """
#     # predicted probabilities based on fitted model
#     proba = fitted_model.predict_proba(X)
    
#     # predicted y based on passed threshold
#     y_pred = [int(i>=threshold) for i in proba[:,1]]

#     # line for nicer output :)
#     print('____________________')

#     # confusion matrix of actual y and predicted y
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         sns.heatmap(cm, cmap='PuOr', annot=True);
#         print('Model Metrics and Normalized Confusion Matrix:')
#         print("_____________________")
#     else:
#         print('Model Metrics and Confusion Matrix without Normalization:')
#         sns.heatmap(cm, cmap='PuOr', annot=True);
#     print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
#     print("_____________________")
#     print('Weighted Quadratic Kappa:', round(cohen_kappa_score(y_true, y_pred, weights='quadratic'), 4)) 
    
    
def model_hyperparams(fitted_model):
  """
  returns set model hyperparams
  required arguments: fitted model
  """
  # return used hyperparameters as df:  
  df_model_params = pd.DataFrame.from_dict(fitted_model.get_params(), orient="index", columns=['set hyperparams'])
  display(df_model_params)



# metrics for cats
def our_metrics_cats(y_true_cats, y_pred_cats, normalize=True): 
    print('**********************************************************************')
    print(f'Weighted Quadratic Kappa for Cats: \
               {round(cohen_kappa_score(y_true_cats, y_pred_cats, weights="quadratic"),4)} \n(Accuracy for Cats: {(round(accuracy_score(y_true_cats, y_pred_cats), 4))})')
    
# metrics for dogs
def our_metrics_dogs(y_true_dogs, y_pred_dogs, normalize=True): 
        print('**********************************************************************')
        print(f'Weighted Quadratic Kappa for Dogs: \
               {round(cohen_kappa_score(y_true_dogs, y_pred_dogs, weights="quadratic"),4)} \n(Accuracy for Dogs: {(round(accuracy_score(y_true_dogs, y_pred_dogs), 4))})')

def comb_metrics(y_true_dogs, y_pred_dogs, y_true_cats, y_pred_cats, normalize=True):
    # conf-matrices dogs and cats
    cm_dogs = confusion_matrix(y_true_dogs, y_pred_dogs)
    cm_cats = confusion_matrix(y_true_cats, y_pred_cats)
    if normalize:
        cm_dogs = cm_dogs.astype('float') / cm_dogs.sum(axis=1)[:, np.newaxis]
        cm_cats = cm_cats.astype('float') / cm_cats.sum(axis=1)[:, np.newaxis]
        fig,ax = plt.subplots(1,2,figsize=(15,5))
        sns.heatmap(ax=ax[0], data=cm_dogs, annot=True)
        sns.heatmap(ax=ax[1], data=cm_cats, annot=True)
        ax[0].set_title('Normalized CM Dogs')
        ax[1].set_title('Normalized CM Cats')
        fig.tight_layout(pad=3)
    else:
        sns.heatmap(ax=ax[0],data=cm_dogs, annot=True)
        sns.heatmap(ax=ax[1], data=cm_cats, annot=True)
        ax[0].set_title('CM Dogs')
        ax[1].set_title('CM Cats')
        fig.tight_layout(pad=3)
    #combined metrics
    kappa_cats = cohen_kappa_score(y_true_cats, y_pred_cats, weights="quadratic")
    accuracy_cats = accuracy_score(y_true_cats, y_pred_cats)
    nr_cats = len(y_true_cats)
    kappa_dogs = cohen_kappa_score(y_true_dogs, y_pred_dogs, weights="quadratic")
    accuracy_dogs = accuracy_score(y_true_dogs, y_pred_dogs)
    nr_dogs = len(y_true_dogs)
    combined_kappa = ((kappa_dogs * nr_dogs) + (kappa_cats * nr_cats)) / (nr_dogs + nr_cats)
    combined_accuracy = ((accuracy_dogs * nr_dogs) + (accuracy_cats * nr_cats)) / (nr_dogs + nr_cats)
    print('**********************************************************************')
    print('**********************************************************************')
    print(f'Combined Kappa: {combined_kappa}\n(Combined Accuracy: {combined_accuracy})')

def cat_dog_metrics(y_true_dogs, y_pred_dogs, y_true_cats, y_pred_cats):
    from sklearn.metrics import cohen_kappa_score
    our_metrics_cats(y_true_cats, y_pred_cats)
    our_metrics_dogs(y_true_dogs, y_pred_dogs)
    comb_metrics(y_true_dogs, y_pred_dogs, y_true_cats, y_pred_cats)

