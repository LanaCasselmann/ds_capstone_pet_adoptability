# import of the necessary libraries
# dataframe and math libraries
import pandas as pd
import numpy as np
# streamlit for building a streamlit web app
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
# serialization of model and scaler
import pickle
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# not needed?
from sklearn.metrics import confusion_matrix
import os



# global plot settings
# set seaborn options globally
colors = ['#365b6d', '#41c1ba', '#289dd2', '#6c9286', '#f2f1ec', '#fa9939']
custom_palette = sns.set_palette(sns.color_palette(colors))
custom_params = {"axes.facecolor": "#f2f1ec", 
"figure.facecolor": "#f2f1ec",
'figure.titleweight': 'bold',
'figure.titlesize': 28,#'large',
'grid.alpha': 1.0,
'font.size': 16.0,
'font.weight': 'bold',
'axes.labelsize': 16,
'axes.labelcolor': '#365b6d',
'axes.titlepad': 10.0,
'axes.titlesize': 'large',
'xtick.labelsize': 16,
'xtick.color': '#365b6d',
'xtick.bottom': True,
'ytick.labelsize': 16,
'ytick.color': '#365b6d',
'ytick.left': True,
'text.color' : '#365b6d',
#'legend.labelcolor': '#365b6d',
'legend.title_fontsize': 12.0,
'legend.frameon': False,
'axes.linewidth': 3,#0.8,
'axes.spines.left': True,
'axes.spines.bottom': True,
'axes.spines.right': True,
'axes.spines.top': True,
'axes.edgecolor': '#365b6d',
'axes.labelweight': 'bold',
'axes.titleweight': 'bold',
'patch.edgecolor': '#f2f1ec'
}
sns.set_theme(style="white", palette=colors, rc=custom_params)


#share and preview settings
#st.set_page_config(page_title="Paw Predictors", page_icon=Image.open('./notebooks/cat_dog_pair.png'))
#st.set_page_config(page_title="Paw Predictors", page_icon=':dog:')
st.set_page_config(layout = "wide", page_title="Paw Predictors", page_icon=Image.open('./notebooks/cat_dog_pair.png'))

# define big font
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
}
</style>
""", unsafe_allow_html=True)

# reduce/set whitespace on top/sides/bottom # left=left of sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2.5rem; 
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)


# df = pd.read_csv("my_data.csv")
# st.line_chart(df)

# import saved serialized model and scaler
loaded_model = pickle.load(open('notebooks/gbc.sav', 'rb'))
loaded_scaler = pickle.load(open('notebooks/scaler.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# -------------------------------------

# setting the headers 

#st.write(os.getcwd())
#ppp = Image.open('./notebooks/ppp_cropped.png')
ppp = Image.open('./notebooks/cat_dog_pair.png')
st.sidebar.image(ppp)

st.write("""
# Paw Predictors
### Predicting the Adoption Speed of Shelter Animals     
""")

# main page with three columns
# in each column: getting user input on pet features
# input data is changed according to feature engineering requirements in order to feed them into the model
col1, col2, col3 = st.columns(3)
with col1: 
    type_in = st.radio(label='##### Type of Animal', options=['Cat', 'Dog'])
    type_bin = 0 if type_in == 'Dog' else 1

    gender_in = st.radio(label='##### Sex of Animal', options=['Male', 'Female'])
    gender_bin = 0 if gender_in == 'Male' else 1

    sterilized_in = st.radio(label='##### Is the Animal sterilized?', options=['No','Yes'])
    sterilized_in_bin = 0 if sterilized_in == 'Yes' else 1

    breed_type_in = st.radio(label='##### Is the Animal pure or mixed breed?', options=['Pure', 'Mixed'])
    breed_type_bin = 0 if breed_type_in == 'Pure' else 1

    vaccinated_dewormed_in = st.radio(label='##### Is the Animal Dewormed and Vaccinated?', options=['Neither','Partly','Fully'])
    vaccinated_dewormed_bin = 0 if vaccinated_dewormed_in == 'Fully' else 1 if vaccinated_dewormed_in == 'Partly' else 2

with col2:

    fee_bin_in = st.radio(label='##### Is an Adoption Fee required?', options=['No', 'Yes'])
    fee_bin_bin = 0 if fee_bin_in == 'No' else 1

    maturitysize_in = st.radio(label='##### Size of Animal at maturity', options=['Small','Medium', 'Large','Extra Large'])
    maturitysize_0_bin = 1 if maturitysize_in == 'Small' else 0
    maturitysize_1_bin = 1 if maturitysize_in == 'Medium' else 0
    maturitysize_2_bin = 1 if maturitysize_in == 'Large' else 0
    maturitysize_3_bin = 1 if maturitysize_in == 'Extra Large' else 0

    furlength_in = st.radio(label='##### Fur length', options=['Short', 'Medium', 'Long'])
    furlength_0_bin = 1 if furlength_in == 'Short' else 0
    furlength_1_bin = 1 if furlength_in == 'Medium' else 0
    furlength_2_bin = 1 if furlength_in == 'Long' else 0

    health_in = st.radio(label='##### Health Condition of Animal', options=['Healthy', 'Minor Injury', 'Serious Injury'])
    health_0_bin = 1 if health_in == 'Healthy' else 0
    health_1_bin = 1 if health_in == 'Minor Injury' else 0
    health_2_bin = 1 if health_in == 'Serious Injury' else 0

    #saved = st.button('Predict',type="primary",use_container_width=True)

with col3:
    color_pattern_in = st.radio(label='##### Color Pattern of Animal', options=['Dark', 'Light', 'Mixed'])
    color_pattern_0_bin = 1 if color_pattern_in == 'Dark' else 0
    color_pattern_1_bin = 1 if color_pattern_in == 'Light' else 0
    color_pattern_2_bin = 1 if color_pattern_in == 'Mixed' else 0

#    photoamt_in = st.slider('##### How many Photos of the Animal are uploaded? If more than 15, please enter 15.', 0, 15, 0)
    photoamt_in = st.slider('##### How many Photos of the Animal are uploaded?', 0, 15, 0)
    #st.write(photoamt_in, 'photos are uploaded.')
    photoamt_11_bin = photoamt_in if photoamt_in <= 11 else 11

#    age_in = st.slider('##### How old is the Animal (in months))? If older than 100 months, please enter 100.', 0, 100, 0)
    age_in = st.slider('##### How old is the Animal (in months)?', 0, 100, 0)
    age_bin_bin = 0 if age_in <= 3 else 1 if age_in <= 12 else 2 if age_in <= 72 else 3
    #st.write('The Animal is ', age_in, 'months old.')
    # newborn: 0-3 months higher adoption speeds up to this age on average (0)
    # puppy/kitten 4-12 (1)
    # adult 13-72 month (2)
    # senior: >= 73 (3)

    description_in = st.text_input('##### Please enter your description text', '')
    #st.text_input.markdown(""" :gray[Please enter your description text]""", 'Animal Description')
    description_char = len(description_in)

    #st.number_input('Please enter a number',0,20)
    #define background of text_input box
    components.html(
        """
    <script>
    const elements = window.parent.document.querySelectorAll('.stTextInput div[data-baseweb="input"] > div')
    console.log(elements)
    elements[0].style.backgroundColor = '#f2f1ec'
    </script>
    """,
        height=0,
        width=0,
    )
    # '#f2f1ec'
dict_input_feat = {'Animal type': type_in,
            'Gender': gender_in ,
            'Sterilized': sterilized_in,
            'Breed': breed_type_in,
            "Dewormed + Vac'ed": vaccinated_dewormed_in,
            'Fee required?': fee_bin_in,
            'Maturity size': maturitysize_in,
            'Fur length': furlength_in,
            'Health condition': health_in,
            'Color pattern': color_pattern_in,
            'No. of photos': photoamt_in,
            'Age': age_in,
            'Description length': description_char
}
#st.sidebar.header(":gray[Your entry:]")

st.sidebar.markdown(''' <p style="color:#f2f1ec",p class="big-font">Your entry:</p>''', unsafe_allow_html=True)
st.sidebar.markdown(f''' <p style="color:#f2f1ec">Animal type: {type_in} <br>
                Gender: {gender_in}  <br>
                Sterilized: {sterilized_in}  <br>
                Breed: {breed_type_in}  <br>
                Dewormed + Vac'ed: {vaccinated_dewormed_in}  <br>
                Fee required?: {fee_bin_in}  <br>
                Maturity size: {maturitysize_in}  <br>
                Fur length: {furlength_in}  <br>
                Health condition: {health_in}  <br>
                Color pattern: {color_pattern_in}  <br>
                No. of photos: {photoamt_in}  <br>
                Age: {age_in}  <br>
                Description length: {description_char}</p>
                ''', unsafe_allow_html=True) 

# st.sidebar.markdown(f""" ##### :gray[
#                 Animal type: {type_in}  
#                 Gender: {gender_in}  
#                 Sterilized: {sterilized_in}  
#                 Breed: {breed_type_in}  
#                 Dewormed + Vac'ed: {vaccinated_dewormed_in}  
#                 Fee required?: {fee_bin_in}  
#                 Maturity size: {maturitysize_in}  
#                 Fur length: {furlength_in}  
#                 Health condition: {health_in}  
#                 Color pattern: {color_pattern_in}  
#                 No. of photos: {photoamt_in}  
#                 Age: {age_in}  
#                 Description length: {description_char}]  
#                """)

# save input data
#saved = st.button('Predict')


# import data for plotting
X_train_comb = pd.read_csv('data/petfinder-adoption-prediction/train/X_train_minmax_scaled_processed.csv')
y_train_comb = pd.read_csv('data/petfinder-adoption-prediction/train/y_train.csv')

df_comb = X_train_comb.copy()
df_comb['adoptionspeed']=y_train_comb

saved = st.button('Predict',type="primary",use_container_width=True)
if saved:
    # save values in dataframe
    d =     {'type' : [type_bin],
            'gender' : [gender_bin],
            'sterilized' : [sterilized_in_bin],
            'breed_type' : [breed_type_bin],
            'vaccinated_dewormed' : [vaccinated_dewormed_bin],
            'fee_bin' : [fee_bin_bin],
            'maturitysize_0' : [maturitysize_0_bin],
            'maturitysize_1' : [maturitysize_1_bin],
            'maturitysize_2' : [maturitysize_2_bin],
            'maturitysize_3' : [maturitysize_3_bin],
            'furlength_0' : [furlength_0_bin],
            'furlength_1' : [furlength_1_bin],
            'furlength_2' : [furlength_2_bin],
            'health_0' : [health_0_bin],
            'health_1' : [health_1_bin],
            'health_2' : [health_2_bin],
            'color_pattern_0' : [color_pattern_0_bin],
            'color_pattern_1' : [color_pattern_1_bin],
            'color_pattern_2' : [color_pattern_2_bin],
            'photoamt_11' : [photoamt_11_bin],
            'age_bin' : [age_bin_bin],
            'description_char' : [description_char]}
    
    df = pd.DataFrame(data=d)
    arr_num_scaled = loaded_scaler.transform(df[['photoamt_11', 'age_bin', 'description_char']]) 
    df_num_scaled = pd.DataFrame(columns=['photoamt_11', 'age_bin', 'description_char'], data=arr_num_scaled)
    df_new = pd.concat([df.drop(['photoamt_11', 'age_bin', 'description_char'], axis=1),df_num_scaled], axis=1)
    
    y_pred = loaded_model.predict(df_new)
    st.write("# Prediction:")
    prediction_string_list = ["The predicted adoption time is < 1 week","The predicted adoption time is between 1 week and 1 month","The predicted adoption time is between 1 and 3 month","The animal will likely not be adopted within 100 days"]
    st.write(f'#### {prediction_string_list[int(y_pred)-1]}')
    #st.write('The predicted Adoption Speed is ', y_pred, '.')
    #st.write(f'The Distribution of Adoption Speeds for {type_in}s')

st.write("""
---
         
### Plot Options
""")
# plot_button = st.button(f'Plot Distribution of Adoption Speeds for {type_in}s')

# if plot_button:
#     fig = plt.figure(figsize=(20,8))
#     speed_plot = sns.histplot(
#     data=df_comb.query('type==@type_bin'), 
#     x='adoptionspeed', stat='proportion', discrete=True,
# #    y = 'accuracy',
#     color='#41c1ba',
#     shrink=.8
#     )
#     plt.xlabel('Adoptionspeed')
#     #plt.ylabel('Accuracy')
#     plt.title(f'The Distribution of Adoption Speeds for {type_in}s')#, fontsize=24)
#     for g in speed_plot.patches:
#         speed_plot.annotate(format(g.get_height(), '.2f'),
#                     (g.get_x() + g.get_width() / 2., g.get_height()),
#                     ha = 'center', va = 'center',
#                     xytext = (0, -20),
#                     textcoords = 'offset points',
#                     color = '#f2f1ec')
#     plt.xticks(ticks=np.linspace(1,4,4))
#     plt.xlim([0.5, 4.5])
#     speed_plot.set_xticklabels(['First Week','First Month','First Three Month','Not Adopted after 100 Days'])
#     # Display the plot in Streamlit
#     st.pyplot(speed_plot.get_figure())
#     #plt.show();

dict_sing_feat = {'Animal type': type_bin,
            'Gender': gender_bin ,
            'Sterilized': sterilized_in_bin,
            'Breed': breed_type_bin,
            "Dewormed + Vac'ed": vaccinated_dewormed_bin,
            'Fee required?': fee_bin_bin,
            'No. of photos': photoamt_11_bin,
            'Age': age_bin_bin,
            'Description length': description_char
}

dict_dummy_feat = {'Maturity size': [maturitysize_0_bin,maturitysize_1_bin,maturitysize_2_bin,maturitysize_3_bin],
            'Fur length': [furlength_0_bin,furlength_1_bin,furlength_2_bin],
            'Health condition': [health_0_bin,health_1_bin,health_2_bin],
            'Color pattern': [color_pattern_0_bin,color_pattern_1_bin,color_pattern_2_bin]
}

feat_choice = st.radio(label='##### Pick a feature to look at its Adoption Speed Distribution', options=list(dict_sing_feat.keys())+ list(dict_dummy_feat.keys()))
plot_button_2 = st.button(f'Plot Distribution of Adoption Speeds for {feat_choice}')
if feat_choice in list(dict_sing_feat.keys()):
    if feat_choice == 'Animal type':
        col_feat_choice = 'type'  
    elif feat_choice == 'Gender':
        col_feat_choice = 'gender'
    elif feat_choice == 'Sterilized':
        col_feat_choice = 'sterilized'
    elif feat_choice == 'Breed':
        col_feat_choice = 'breed_type'
    elif feat_choice == "Dewormed + Vac'ed":
        col_feat_choice = 'vaccinated_dewormed'
    elif feat_choice == 'Fee required?':
        col_feat_choice = 'fee_bin'
    elif feat_choice == 'No. of photos':
        col_feat_choice = 'photoamt_11'
    elif feat_choice == 'Age':
        col_feat_choice = 'age_bin'
    elif feat_choice == 'Description length': 
        col_feat_choice = 'description_char'
    feat_choice_value = dict_sing_feat[feat_choice]
    query_str = f'{col_feat_choice}=={feat_choice_value}'

if feat_choice in list(dict_dummy_feat.keys()):
    if feat_choice == 'Maturity size':
        col_feat_choice = ['maturitysize_0','maturitysize_1','maturitysize_2','maturitysize_3']
    elif feat_choice == 'Fur length':
        col_feat_choice = ['furlength_0','furlength_1','furlength_2']
    elif feat_choice == 'Health condition':
        col_feat_choice = ['health_0','health_1','health_2']
    elif feat_choice == 'Color pattern':
        col_feat_choice = ['color_pattern_0','color_pattern_1','color_pattern_2']
    feat_choice_value = dict_dummy_feat[feat_choice]
    query_str = ""
    for col,val in zip(col_feat_choice, feat_choice_value):
        query_str = query_str + f'{col}=={val} and '
    query_str = query_str[:-5]

if plot_button_2:
    fig = plt.figure(figsize=(20,8))
    speed_plot_2 = sns.histplot(
    data=df_comb.query(query_str), 
    x='adoptionspeed', stat='proportion', discrete=True,
#    y = 'accuracy',
    color='#41c1ba',
    shrink=.8
    )
    plt.xlabel('Adoptionspeed')
    #plt.ylabel('Accuracy')
    plt.title(f'The Distribution of Adoption Speeds for your choice of {feat_choice}: {dict_input_feat[feat_choice]}')#, fontsize=24)
    for g in speed_plot_2.patches:
        speed_plot_2.annotate(format(g.get_height(), '.2f'),
                    (g.get_x() + g.get_width() / 2., g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, -20),
                    textcoords = 'offset points',
                    color = '#f2f1ec')
    plt.xticks(ticks=np.linspace(1,4,4))
    plt.xlim([0.5, 4.5])
    speed_plot_2.set_xticklabels(['First Week','First Month','First Three Month','Not Adopted after 100 Days'])
    # Display the plot in Streamlit
    st.pyplot(speed_plot_2.get_figure())
    #plt.show();