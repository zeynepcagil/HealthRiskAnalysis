import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Attack Risk Prediction", layout="centered")

st.title("Heart Attack Prediction Application")

model = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.header("Write the information of patient")

#Numerical Inputs
age = st.sidebar.slider("Age", 20, 80, 45)
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 240)
thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)

# Categorical Inputs
sex = st.sidebar.selectbox("Sex", ["Women", "Man"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"])
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.selectbox("ECD Result", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
slope = st.sidebar.selectbox("Slope Type", ["Type 0", "Type 1", "Type 2"])
ca = st.sidebar.selectbox("Number of Veins Seen with Fluoroscopy", ["0", "1", "2", "3", "4"])
thal = st.sidebar.selectbox("Thalium Scan", ["0", "1", "2", "3"])

# The Predict Button
if st.sidebar.button("PREDICT"):
    numeric_input = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak
    }

    input_dict = {
        'sex_1': 1 if sex == "Man" else 0,
        'cp_1': 1 if cp == "Type 1" else 0,
        'cp_2': 1 if cp == "Type 2" else 0,
        'cp_3': 1 if cp == "Type 3" else 0,
        'fbs_1': 1 if fbs == "Yes" else 0,
        'restecg_1': 1 if restecg == "ST-T anormality" else 0,
        'restecg_2': 1 if restecg == "Left ventricular hypertrophy" else 0,
        'exang_1': 1 if exang == "Yes" else 0,
        'slope_1': 1 if slope == "Type 1" else 0,
        'slope_2': 1 if slope == "Type 2" else 0,
        'ca_1': 1 if ca == "1" else 0,
        'ca_2': 1 if ca == "2" else 0,
        'ca_3': 1 if ca == "3" else 0,
        'ca_4': 1 if ca == "4" else 0,
        'thal_1': 1 if thal == "1" else 0,
        'thal_2': 1 if thal == "2" else 0,
        'thal_3': 1 if thal == "3" else 0
    }

    all_inputs = {**numeric_input, **input_dict}
    df_input = pd.DataFrame([all_inputs])
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df_input[numeric_features] = scaler.transform(df_input[numeric_features])

    prediction = model.predict(df_input)

    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error("High Risk. Please see a doctor.")
        labels = ['At Risk', 'Safe']
        sizes = [70, 30]
        colors = ['#ff4d4d', '#90ee90']
    else:
        st.success("Low Risk. Take care of your health!")
        labels = ['Safe', 'At Risk']
        sizes = [85, 15]
        colors = ['#90ee90', '#ff9999']

    # Pie Graph
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")
