import streamlit as st
import pandas as pd
from joblib import load

# Load the trained pipeline (model + preprocessing)
@st.cache_resource
def load_model():
    return load("car_price_model.joblib")  # Make sure this file is in the same folder

model = load_model()

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗")
st.title("🚗 Used Car Price Prediction")
st.write("Enter car details below and click **Predict** to estimate the selling price.")

# --- User Inputs ---
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2017)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.59, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=27000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# --- Prepare Input Data ---
input_data = pd.DataFrame([{
    "year": year,
    "present_price": present_price,
    "kms_driven": kms_driven,
    "fuel_type": fuel_type,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner
}])

# --- Predict Button ---
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Selling Price: **{prediction[0]:.2f} Lakhs**")
