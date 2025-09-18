#!/usr/bin/env python3
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model
@st.cache_resource
def load_model():
    with open("best_model_GradientBoosting.pickle", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write("Enter car details below to predict the price.")

# --- Replace these with your actual feature names ---
numeric_features = ["year", "mileage"]   # Example numeric features
categorical_features = ["fuel_type", "transmission"]  # Example categorical features

# Collect user input
num_inputs = {}
for feature in numeric_features:
    num_inputs[feature] = st.number_input(f"Enter {feature}", min_value=0, value=0)

cat_inputs = {}
for feature in categorical_features:
    cat_inputs[feature] = st.selectbox(f"Select {feature}", ["Option1", "Option2", "Option3"])

# Convert input to DataFrame
input_data = {**num_inputs, **cat_inputs}
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Price: {prediction:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("⚠️ Please make sure your feature names and options match the training dataset.")
