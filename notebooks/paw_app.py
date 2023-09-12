import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os
#st.write(os.getcwd())
ppp = Image.open('./notebooks/ppp.jpg')
st.image(ppp)

st.write("""
# Paw Predictors
Predicting the Adoption Speed of Shelter Animals
""")

# df = pd.read_csv("my_data.csv")
# st.line_chart(df)

# import saved model
# load the model from disk
loaded_model = pickle.load(open('notebooks/gbc.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

#st.write('#### Type of Animal')
type_in = st.radio(label='##### Type of Animal', options=['Cat', 'Dog'])
type_bin = 0 if type_in == 'Dog' else 1

gender_in = st.radio(label='#### Sex of Animal', options=['Male', 'Female'])
gender_bin = 0 if gender_in == 'Male' else 1

sterilized_in = st.radio(label='#### Is the Animal sterilized?', options=['Yes', 'No'])
sterilized_in_bin = 0 if sterilized_in == 'Yes' else 1

breed_type_in = st.radio(label='#### Is the Animal pure or mixed breed?', options=['Pure', 'Mixed'])
breed_type_bin = 0 if breed_type_in == 'Pure' else 1

vaccinated_dewormed_in = st.radio(label='#### Is the Animal Dewormed and Vaccinated?', options=['Fully', 'Partly', 'Neither'])
vaccinated_dewormed_bin = 0 if vaccinated_dewormed_in == 'Fully' else 1 if vaccinated_dewormed_in == 'Partly' else 2

fee_bin_in = st.radio(label='#### Is an Adoption Fee required?', options=['Yes', 'No'])
fee_bin_bin = 0 if fee_bin_in == 'No' else 1

maturitysize_in = st.radio(label='#### Size of Animal at maturity', options=['Small','Medium', 'Large','Extra Large'])
maturitysize_0_bin = 1 if maturitysize_in == 'Small' else 0
maturitysize_1_bin = 1 if maturitysize_in == 'Medium' else 0
maturitysize_2_bin = 1 if maturitysize_in == 'Large' else 0
maturitysize_3_bin = 1 if maturitysize_in == 'Extra Large' else 0

furlength_in = st.radio(label='#### Fur length', options=['Short', 'Medium', 'Long'])
furlength_0_bin = 1 if furlength_in == 'Short' else 0
furlength_1_bin = 1 if furlength_in == 'Medium' else 0
furlength_2_bin = 1 if furlength_in == 'Long' else 0

health_in = st.radio(label='#### Health Condition of Animal', options=['Healthy', 'Minor Injury', 'Serious Injury'])
health_0_bin = 1 if health_in == 'Healthy' else 0
health_1_bin = 1 if health_in == 'Minor Injury' else 0
health_2_bin = 1 if health_in == 'Serious Injury' else 0

color_pattern_in = st.radio(label='#### Color Pattern of Animal', options=['Dark', 'Mixed', 'Light'])
color_pattern_0_bin = 1 if color_pattern_in == 'Dark' else 0
color_pattern_1_bin = 1 if color_pattern_in == 'Light' else 0
color_pattern_2_bin = 1 if color_pattern_in == 'Mixed' else 0

photoamt_in = st.slider('#### How many Photos of the Animal are uploaded?', 0, 20, 1)
st.write(photoamt_in, 'photos are uploaded.')
photoamt_11_bin = photoamt_in if photoamt_in <= 11 else 11

age_in = st.slider('#### How old is the Animal (in months))?', 0, 300, 1)
age_bin_bin = 0 if age_in <= 3 else 1 if age_in <= 12 else 2 if age_in <= 72 else 3
st.write('The Animal is ', age_in, 'months old.')
# newborn: 0-3 months higher adoption speeds up to this age on average (0)
# puppy/kitten 4-12 (1)
# adult 13-72 month (2)
# senior: >= 73 (3)

description_in = st.text_input('#### Please enter your description text', 'Animal Description')
description_char = len(description_in)

# save input data
saved = st.button('Predict')

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
    df_new = pd.DataFrame(data=d)
    y_pred = loaded_model.predict(df_new)
    st.write("# Prediction:")
    prediction_string_list = ["The predicted adoption time is < 1 week","The predicted adoption time is between 1 week and 1 month","The predicted adoption time is between 1 and 3 month","The animal will likely not be adopted within 100 days"]
    st.write(f'{prediction_string_list[int(y_pred)]}')
    #st.write('The predicted Adoption Speed is ', y_pred, '.')