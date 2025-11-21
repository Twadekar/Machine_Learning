import streamlit as st
import pandas as pd
import numpy as np
import pickle as pickle

#Load the trained model
with open("logistic_model.pkl","rb") as file:
    model = pickle.load(file)

#App title
st.title("Titanic Survival Prediction")

#Description
st.write("This app predicts wheter a Titanic passenger would survive based on their details.")

#Input fields for user data
st.header("Passenger Information")



pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)",[1,2,3])
sex = st.selectbox("Sex (0=Male,1=Female)",[0,1])
age = st.slider("Age", min_value=1,max_value=100,value=30)
sibsp = st.slider("Number of siblings/Spouses Abroad", min_value=0,max_value=8,value=0)
parch = st.slider("Number of Parents/Children Abroad",min_value=0,max_value=9,value=0)
fare = st.slider("Fare Paid",min_value=0.0,max_value=550.0, value=32.0)
embarked_s= st.selectbox("Embarked at Southampton (1 = Yes, 0 = No)",[0,1])
embarked_q = st.selectbox("embarked at Queenstown (1 = Yes, 0 = No)",[0,1])

#Prepare input for prediction
embarked_c = 1 if embarked_s ==0 and embarked_q ==0 else 0 
input_data = np.array([[pclass,sex,age,sibsp,parch,fare,embarked_s,embarked_q]])

#Make predictions
if st.button("Predict survival"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction[0]==1:
        st.success(f"The passenger is predicted to survive witha probablity of {prediction_proba:.2f}.")
    else:
        st.error(f"The passenger is predicted not to survive with a probability of {1 - prediction_proba:.2f}.")
