#!/usr/bin/env python3
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open("best_model_GradientBoosting.pickle", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------
# Define your features here
# Replace with your actual column names
numeric_features = ["year", "mileage"]   # Example numeric inputs
categorical_features = ["fuel_type", "transmission"]  # Example categorical inputs
# -------------------------------

st.title("🚗 Used Car Price Prediction App")
st.write("Enter car details below and get the predicted price.")

# Input form
with st.form("prediction_form"):
    # Numeric inputs
    numeric_data = {}
    for col in numeric_features:
        numeric_data[col] = st.number_input(f"Enter {col}", min_value=0, step=1)

    # Categorical inputs
    categorical_data = {}
    for col in categorical_features:
        categorical_data[col] = st.selectbox(f"Select {col}", ["Option1", "Option2", "Option3"])

    submit = st.form_submit_button("Predict Price")

if submit:
    # Combine numeric + categorical into DataFrame
    input_data = {**numeric_data, **categorical_data}
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Price: {prediction:,.2f}")
