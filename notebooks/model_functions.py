
# for display dataframe

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score

def model_hyperparams(fitted_model):
  """
  returns set model hyperparamenters
  required arguments: fitted model
  """
  # return used hyperparameters as df:  
  df_model_params = pd.DataFrame.from_dict(fitted_model.get_params(), orient="index", columns=['set hyperparameters'])
  # display df:
  display(df_model_params)


def metrics_threshold(fitted_model, X_test, y_test, threshold=0.5, normalize=True):
    """
    function to compute confusion matrix and classification metrics based on passed threshold.
    required arguments: fitted model, X_test, y_test, threshold (float, default >= 0.5)
    """
    # predicted probabilities based on fitted model
    proba = fitted_model.predict_proba(X_test)
    # predicted y based on passed threshold
    y_pred = [int(i>=threshold) for i in proba[:,1]]
    # line for nicer output :)
    print('____________________')
    # confusion matrix of actual y and predicted y
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, annot=True);
        print('Model Metrics and Normalized Confusion Matrix')
        print("_____________________")
        print("_____________________")
    else:
        print('Model Metrics and Confusion Matrix without Normalization')
        print("_____________________")
        print("_____________________")
        sns.heatmap(cm, annot=True);
    # accuracy score of confusion matrix
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("_____________________")
    print('Weighted Quadratic Kappa:', round(cohen_kappa_score(y_test, y_pred, weights='quadratic'), 4)) 