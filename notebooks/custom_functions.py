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
    