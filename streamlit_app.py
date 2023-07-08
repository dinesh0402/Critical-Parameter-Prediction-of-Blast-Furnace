# App to host the Application online.

# Import necessary libraries.
import streamlit as st
import numpy as np
import pandas as pd
import zipfile
import joblib
from sklearn.preprocessing import StandardScaler


# Path to the zip file
zip_file_path = "models.zip"

# Name of the pickle file inside the zip
pickle_file_name_1 = "mlr_model.pkl"
pickle_file_name_2 = "rfr_model.pkl"
pickle_file_name_3 = "dnn_model.pkl"
pickle_file_name_4 = "sample_model.pkl"


# Prediction funtion
# It uses already trained learning models to predict the critical parameters.
def predict(model_no, columns, row): 
    
    # Scale the data.
    scaler = StandardScaler()
    
    df = pd.read_csv('modified_bf_data.csv', index_col=[0])
    X = df.drop(['SAT_1','SAT_2','SAT_3','SAT_4'],axis=1)
    X.drop('DATE_TIME',axis=1,inplace=True)
    y = df[['SAT_1','SAT_2','SAT_3','SAT_4']]
    X = scaler.fit_transform(X)
    
    # Select the model.
    if(model_no == 1):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            # Read the pickle file from the zip
            with zip_file.open(pickle_file_name_1) as pickle_file:
                # Load the pickle data
                model = joblib.load(pickle_file)
    elif(model_no == 2):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            # Read the pickle file from the zip
            with zip_file.open(pickle_file_name_2) as pickle_file:
                # Load the pickle data
                model = joblib.load(pickle_file)
    else:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            # Read the pickle file from the zip
            with zip_file.open(pickle_file_name_3) as pickle_file:
                # Load the pickle data
                model = joblib.load(pickle_file)
    
    # Prepare the feature vector.
    row = np.array([cb_flow,cb_press,cb_temp,steam_flow,steam_temp,steam_press,o2_press,o2_flow,o2_per,pci,atm_humid,hb_temp,hb_press,top_press,top_temp1,top_temp2,top_temp3,top_temp4,top_spray,top_temp,top_press1,co,co2,h2,skin_avg_temp]) 
    
    # Uncomment for testing purposes...
    # row = np.array([311727,3.15,129,4,213,3.34,3.2,7296,23.08,32,24.56,1060,2.99,1.5,112,135,107,130,0,121,2,22.22,21,3.88,69.940478])
    
    # Transform the Feature Vector.
    row = scaler.transform([row])
    X_new = pd.DataFrame(row, columns = columns)
    prediction = model.predict(X_new)
    
    # Output the predictions.
    st.markdown(
        f""" 
            <style>
            .element-container:has(iframe[height="0"]) {{
            display: none;
            }}
            </style>
        """, unsafe_allow_html=True
    )
    
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''

    st.components.v1.html(js, height=0)
    
    st.header('The Predictions for the next 4 hours are :')
    st.write('Average Skin Temperature after 1 Hour : ', prediction[0][0],'ºC')
    st.write('Average Skin Temperature after 2 Hours : ', prediction[0][1],'ºC')
    st.write('Average Skin Temperature after 3 Hours : ', prediction[0][2],'ºC')
    st.write('Average Skin Temperature after 4 Hours : ', prediction[0][3],'ºC')

    


# Input the model.
with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            # Read the pickle file from the zip
            with zip_file.open(pickle_file_name_4) as pickle_file:
                # Load the pickle data
                model = joblib.load(pickle_file)
cols = pd.read_csv('columns.csv')
columns = cols['Params'].to_list()

# Description of the project.
st.header('Predicting The Temperature Parameters of Blast Furnace')
st.markdown('In this app, you can provide the necessary parameters as input and obtain the predictions for the "Average Skin Temperature" over the course of next 4 hours !!!')


# Parameters Input.

cb_flow = st.number_input('Cold Blast Flow')
cb_press = st.number_input('Cold Blast Pressure')
cb_temp = st.number_input('Cold Blast Temperature')
steam_flow = st.number_input('Steam Flow')
steam_temp = st.number_input('Steam Temperature')
steam_press = st.number_input('Steam Pressure')
o2_press = st.number_input('O2 Pressure')
o2_flow = st.number_input('O2 Flow')
o2_per = st.number_input('O2 Percentage')
pci = st.number_input('Pulverized Coal Injection')
atm_humid = st.number_input('Atmospheric Humidity')
hb_temp = st.number_input('Hot Blast Temperature')
hb_press = st.number_input('Hot Blast Pressure')
top_press = st.number_input('Top Gas Pressure')
top_temp1 = st.number_input('Top Gas Temperature after 10 minutes')
top_temp2 = st.number_input('Top Gas Temperature after 20 minutes')
top_temp3 = st.number_input('Top Gas Temperature after 30 minutes')
top_temp4 = st.number_input('Top Gas Temperature after 40 minutes')
top_spray = st.number_input('Top Gas Spray')
top_temp = st.number_input('Top Gas Temperature')
top_press1 = st.number_input('Top Gas Pressure 1')                           
co = st.number_input('CO (Carbon Monoxide)')
co2 = st.number_input('CO2 (Carbon Dioxide)')
h2 = st.number_input('H2 (Hydrogen)')
skin_avg_temp = st.number_input('Average Skin Temperature')

row = np.array([cb_flow,cb_press,cb_temp,steam_flow,steam_temp,steam_press,o2_press,o2_flow,o2_per,pci,atm_humid,hb_temp,hb_press,top_press,top_temp1,top_temp2,top_temp3,top_temp4,top_spray,top_temp,top_press1,co,co2,h2,skin_avg_temp]) 

      
# Select the learning algorithm.
lr_alg = st.selectbox(
    'Select the learning algorithm :',
    ('Multiple Linear Regression', 'Random Forest Regression', 'Deep Neural Network Regression'), index=2)

lr_map = {'Multiple Linear Regression':1, 'Random Forest Regression':2, 'Deep Neural Network Regression':3}

# Predict function triggered.
trigger = st.button('Predict', on_click=predict, args=(lr_map[lr_alg],columns,row))