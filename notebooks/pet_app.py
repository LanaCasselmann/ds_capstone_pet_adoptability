from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import pandas as pd
import numpy as np
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import dexplot as dxp

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
# import graphviz
# Scaling with Minmax-scaler
from sklearn.preprocessing import MinMaxScaler

# from ydata_profiling import ProfileReport

# import custom functions
from custom_functions import our_metrics

# import features for tree-based models
X_train_comb = pd.read_csv('~/neuefische/ds_capstone_pet_adoptability/data/petfinder-adoption-prediction/train/X_train_minmax_scaled_processed.csv')
#                           ../data/petfinder-adoption-prediction/train/X_train_minmax_scaled_processed.csv')
X_test_comb =pd.read_csv('~/neuefische/ds_capstone_pet_adoptability/data/petfinder-adoption-prediction/train/X_test_minmax_scaled_processed.csv')
# import target
y_train_comb = pd.read_csv('~/neuefische/ds_capstone_pet_adoptability/data/petfinder-adoption-prediction/train/y_train.csv')
y_test_comb = pd.read_csv('~/neuefische/ds_capstone_pet_adoptability/data/petfinder-adoption-prediction/train/y_test.csv')
# initialize best model
gbc = GradientBoostingClassifier(n_estimators=200,subsample=1, max_leaf_nodes=31, max_features='log2', max_depth=5, loss = 'log_loss', learning_rate=0.025, random_state=42)
gbc.fit(X_train_comb,y_train_comb)
# Performance on test
y_pred = gbc.predict(X_test_comb)
#our_metrics(y_test_comb,y_pred)
# Performance on train
y_pred_tr = gbc.predict(X_train_comb)
#our_metrics(y_train_comb,y_pred_tr)
# Preliminaries for Baseline Model
df_processed = pd.read_csv('~/neuefische/ds_capstone_pet_adoptability/data/petfinder-adoption-prediction/train/df_processed.csv')

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Paw Predictors', style={'textAlign':'center'}),
    dcc.Dropdown(df_processed.type.unique(), id='dropdown-selection'),#'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df_processed[df_processed.type==value]
    return px.scatter(dff, x='age_bin', y='adoptionspeed') # line(dff, x='year', y='pop')

if __name__ == '__main__':
    app.run(debug=True)

# from dash import Dash, html

# app = Dash(__name__)

# app.layout = html.Div([
#     html.Div(children='Paw Predictors')
# ])

# if __name__ == '__main__':
#     app.run(debug=True)