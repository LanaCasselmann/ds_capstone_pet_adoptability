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
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
        print('Model Metrics and Normalized Confusion Matrix')
        print("_____________________")
        print("_____________________")
    else:
        print('Model Metrics and Confusion Matrix without Normalization')
        print("_____________________")
        print("_____________________")
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("_____________________")
    print('Weighted Quadratic Kappa:', round(cohen_kappa_score(y_true, y_pred, weights='quadratic'), 4)) 
    
    
# func to evaluate cat and dog models

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
    """Gives accuracy and quadratic kappa score as well as confusion matrix of the entered cats and dogs data set as well as of the combined data.

    Args:
        y_true_dogs (array_like): true target observations for dogs in data set
        y_pred_dogs (array_like): predicted target observations for dogs in data set
        y_true_cats (array_like): true target observations for cats in data set
        y_pred_cats (array_like): predicted target observations for cats in data set
        normalize (bool, optional): determines whether the confusion matrices are returned with absolute values or normalized . Defaults to True.
    """
    # conf-matrices dogs and cats
    cm_dogs = confusion_matrix(y_true_dogs, y_pred_dogs)
    cm_cats = confusion_matrix(y_true_cats, y_pred_cats)
    cm_comb = np.add(cm_cats, cm_dogs)
    if normalize:
        cm_dogs = cm_dogs.astype('float') / cm_dogs.sum(axis=1)[:, np.newaxis]
        cm_cats = cm_cats.astype('float') / cm_cats.sum(axis=1)[:, np.newaxis]
        cm_comb = cm_comb.astype('float') / cm_comb.sum(axis=1)[:, np.newaxis]
        fig,ax = plt.subplots(1,3,figsize=(15,5))
        sns.heatmap(ax=ax[0], data=cm_dogs, annot=True)
        sns.heatmap(ax=ax[1], data=cm_cats, annot=True)
        sns.heatmap(ax=ax[2], data=cm_comb, annot=True)
        ax[0].set_title('Normalized CM Dogs')
        ax[1].set_title('Normalized CM Cats')
        ax[2].set_title('Normalized CM Combined Set')
        fig.tight_layout(pad=3)
    else:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
        sns.heatmap(ax=ax[0],data=cm_dogs, annot=True)
        sns.heatmap(ax=ax[1], data=cm_cats, annot=True)
        sns.heatmap(ax=ax[1], data=cm_comb, annot=True)
        ax[0].set_title('CM Dogs')
        ax[1].set_title('CM Cats')
        ax[2].set_title('CM Combined Set')
        fig.tight_layout(pad=3)
    #combined metrics
    kappa_cats = cohen_kappa_score(y_true_cats, y_pred_cats, weights="quadratic")
    accuracy_cats = accuracy_score(y_true_cats, y_pred_cats)
    nr_cats = len(y_true_cats)
    kappa_dogs = cohen_kappa_score(y_true_dogs, y_pred_dogs, weights="quadratic")
    accuracy_dogs = accuracy_score(y_true_dogs, y_pred_dogs)
    nr_dogs = len(y_true_dogs)
    y_true_comb = np.concatenate([y_true_cats, y_true_dogs])
    y_pred_comb = np.concatenate([y_pred_cats, y_pred_dogs])
    #combined_kappa = ((kappa_dogs * nr_dogs) + (kappa_cats * nr_cats)) / (nr_dogs + nr_cats)
    #combined_accuracy = ((accuracy_dogs * nr_dogs) + (accuracy_cats * nr_cats)) / (nr_dogs + nr_cats)
    print('**********************************************************************')
    print('**********************************************************************')
    #print(f'Combined Kappa: {combined_kappa}\n(Combined Accuracy: {combined_accuracy})')
    print(f'Combined Kappa: {cohen_kappa_score(y_true_comb, y_pred_comb, weights="quadratic")}\n(Combined Accuracy: {accuracy_score(y_true_comb, y_pred_comb)})')

def cat_dog_metrics(y_true_dogs, y_pred_dogs, y_true_cats, y_pred_cats):
    from sklearn.metrics import cohen_kappa_score
    our_metrics_cats(y_true_cats, y_pred_cats)
    our_metrics_dogs(y_true_dogs, y_pred_dogs)
    comb_metrics(y_true_dogs, y_pred_dogs, y_true_cats, y_pred_cats)