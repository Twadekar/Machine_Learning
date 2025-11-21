import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib



# Load trained model
with open("model.pkl", "rb") as p:
    model = pickle.load(p)
    features = joblib.load("features.pkl")

st.title("Heart Disease Prediction")
st.write("Please enter required values to predict heart disease")


# Input fields
age = st.number_input("Enter age", min_value=0, max_value=120, step=1)

sex = st.selectbox("Sex", ("Male", "Female"))
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest pain type", ["typical", "atypical", "non-anginal"])
cp_typical_angina = 1 if cp == "typical" else 0
cp_atypical_angina = 1 if cp == "atypical" else 0
cp_non_anginal = 1 if cp == "non-anginal" else 0

trestbps = st.number_input("Enter resting blood pressure", min_value=0)

chol = st.number_input("Enter cholesterol", min_value=0)

fbs_True = st.selectbox("Fasting blood sugar > 120 mg/dl", ("Yes", "No"))
fbs_True = 1 if fbs_True == "Yes" else 0

restecg = st.selectbox("Resting ECG", ["normal", "lv_hypertrophy" ,"ST abnormality"])
restecg_normal = 1 if restecg == "normal" else 0
restecg_lv_hypertrophy = 1 if restecg == "normal" else 0
restecg_st_t_abnormality = 1 if restecg == "ST abnormality" else 0

thalch = st.number_input("Enter max heart rate achieved", min_value=0)

exang = st.selectbox("Exercise induced angina", ("Yes", "No"))
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("Enter oldpeak", format="%.2f")

slope = st.selectbox("Slope of peak exercise ST segment", ["flat", 'downsloping', "upsloping"])
slope_flat = 1 if slope == "flat" else 0
slope_downsloping = 1 if slope == "flat" else 0
slope_upsloping = 1 if slope == "upsloping" else 0

thal = st.selectbox("Thalassemia", ["normal", 'fixed_defect', "reversible defect"])
thal_fixed_defect = 1 if thal == "normal" else 0
thal_normal = 1 if thal == "normal" else 0
thal_reversable_defect = 1 if thal == "reversible defect" else 0


input_data = pd.DataFrame([["age", "sex", "trestbps", "chol", "thalch", "exang", "oldpeak",
       "cp_atypical_angina", "cp_non_anginal", "cp_typical_angina", "fbs_True",
       "restecg_normal", "restecg_st_t_abnormality", "slope_flat",
       "slope_upsloping", "thal_normal", "thal_reversable_defect"]], columns=features)


numeric_cols = ['age', 'sex', "trestbps", "chol", "thalch", "exang", "oldpeak"]
for col in numeric_cols:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

input_data = input_data.astype(float)
st.write("Dtypes", input_data.dtypes)
st.write('input data')


# Predict button
if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("⚠️Person is likely to have heart disease.")
    else:
        st.success("✅Person is unlikely to have heart disease.")


