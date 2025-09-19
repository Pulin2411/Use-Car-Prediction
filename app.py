import pickle
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Page setup
st.set_page_config(page_title="Used Car Price Predictor - By Pulin", page_icon="🚗", layout="wide")

st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
}

/* Hero Section */
.hero {
  padding: 24px;
  text-align: center;
  background: linear-gradient(90deg, #007bff, #00c6ff);
  color: white;
  border-radius: 16px;
  margin-bottom: 20px;
}
.hero h1 { font-size: 2rem; margin-bottom: 0; }
.hero p { font-size: 1rem; margin-top: 6px; }

/* Card Style */
.card {
  background: white;
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  margin-bottom: 20px;
}
.result-box {
  padding: 20px;
  border-radius: 14px;
  background: #f9fafc;
  text-align: center;
  font-size: 1.3rem;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
  <h1>🚗 Used Car Price Predictor</h1>
  <p>By Pulin — Enter details and get instant predictions</p>
</div>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = "best_model_GradientBoosting.pickle"
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
model = load_model(MODEL_PATH)

# Dynamic car list
def get_car_options(model):
    if not isinstance(model, Pipeline):
        return []
    for _, step in model.steps:
        if isinstance(step, ColumnTransformer):
            preproc = step
            for _, transformer, cols in preproc.transformers:
                enc = transformer
                if isinstance(transformer, Pipeline) and transformer.steps:
                    enc = transformer.steps[0][1]
                if isinstance(enc, OneHotEncoder) and hasattr(enc, "categories_"):
                    for cats, col in zip(enc.categories_, cols):
                        if str(col).lower() in ["car_name", "name", "model", "car_model", "brand_model"]:
                            return sorted([str(x) for x in cats if str(x).strip() != ""])
    return []

car_options = get_car_options(model) or ["Maruti Swift", "Hyundai i20", "Honda City", "Toyota Innova"]

# Layout: Two columns
left, right = st.columns([1.3, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Car Details")
    car_name = st.selectbox("Car Name", car_options)
    year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=500)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
    predict = st.button("🔍 Predict Price")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    result_container = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

if predict:
    input_data = pd.DataFrame([[car_name, year, km_driven, fuel, seller_type, transmission, owner]],
                              columns=["Car_Name", "Year", "KM_Driven", "Fuel", "Seller_Type", "Transmission", "Owner"])
    prediction = model.predict(input_data)[0]
    result_container.markdown(f'<div class="result-box">Predicted Price: ₹ {prediction:,.2f}</div>', unsafe_allow_html=True)
