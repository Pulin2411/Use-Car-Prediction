import pickle
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Used Car Price Predictor - By Pulin", page_icon="🚗", layout="centered")

st.markdown("""
<style>
body { background: #f8fafc; }

/* Centered card */
.form-card {
  background: white;
  padding: 30px;
  border-radius: 18px;
  box-shadow: 0 6px 16px rgba(0,0,0,.08);
  max-width: 420px;
  margin: auto;
}

.title { text-align: center; font-size: 1.8rem; margin-bottom: 20px; font-weight: 700; }
.result-box {
  padding: 18px;
  border-radius: 12px;
  background: #eef4ff;
  text-align: center;
  font-size: 1.2rem;
  font-weight: 600;
  margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='form-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🚗 Used Car Price Predictor<br><small>By Pulin</small></div>", unsafe_allow_html=True)

MODEL_PATH = "best_model_GradientBoosting.pickle"
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
model = load_model(MODEL_PATH)

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

car_name = st.selectbox("Car Name", car_options)
year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=500)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])

predict = st.button("🔍 Predict Price")

if predict:
    input_data = pd.DataFrame([[car_name, year, km_driven, fuel, seller_type, transmission, owner]],
                              columns=["Car_Name", "Year", "KM_Driven", "Fuel", "Seller_Type", "Transmission", "Owner"])
    prediction = model.predict(input_data)[0]
    st.markdown(f'<div class="result-box">Predicted Price: ₹ {prediction:,.2f}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
