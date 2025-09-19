import pickle
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- Numpy unpickling shim ----
try:
    import numpy as _np
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
except Exception:
    pass

# ---- Page setup ----
st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="centered")

st.markdown("""
<style>
/* Background with light/dark mode */
.stApp {
  background: var(--background-color);
  color: var(--text-color);
}

:root {
  --background-color: #ffffff;
  --card-bg: #ffffff;
  --border-color: #e5eaf0;
  --text-color: #000000;
  --result-bg: #f0f4fa;
  --button-bg: #007bff;
  --button-hover: #0056b3;
}

@media (prefers-color-scheme: dark) {
  :root {
    --background-color: #0e1117;
    --card-bg: #1e222d;
    --border-color: #2c2f36;
    --text-color: #f5f5f5;
    --result-bg: #2c313c;
    --button-bg: #1f6feb;
    --button-hover: #1158c7;
  }
}

/* Mobile-friendly container */
.block-container {
  max-width: 480px;
  padding-top: 1rem;
  padding-bottom: 2rem;
  margin: auto;
}

/* Sidebar */
.css-1d391kg { padding-top: 2rem; }

/* Cards */
.card {
  border-radius: 14px;
  padding: 16px 16px 8px 16px;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  box-shadow: 0 6px 14px rgba(0,0,0,.05);
  margin-bottom: 16px;
}
.card-title { font-weight:600; margin-bottom: 8px; font-size: 1.1rem; }

/* Buttons */
.stButton>button {
  width: 100%;
  border-radius: 12px;
  padding: 12px;
  font-size: 1rem;
  background-color: var(--button-bg);
  color: white;
}
.stButton>button:hover {
  background-color: var(--button-hover);
}

/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div {
  border-radius: 10px;
  font-size: 15px;
  padding: 8px;
  color: var(--text-color);
}

/* Result */
.result-box {
  padding: 16px;
  border-radius: 12px;
  background: var(--result-bg);
  text-align: center;
  font-size: 1.2rem;
  font-weight: 600;
  margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    "This app predicts **used car prices** based on details you provide.\n\n"
    "- Enter car features on the main page.\n"
    "- Click **Predict** to see the estimated price.\n\n"
    "Built with **Streamlit** and an ML model."
)

st.sidebar.header("📘 Instructions")
st.sidebar.markdown("""
1. Fill in vehicle details.
2. Select listing & ownership info.
3. Click **Predict Price**.
4. Check result instantly.
""")

st.title("🚗 Used Car Price Predictor")

# Load the model
MODEL_PATH = "best_model_GradientBoosting.pickle"
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(MODEL_PATH)

# ---- Form Layout ----
with st.form("prediction_form", clear_on_submit=False):
    st.subheader("Enter Car Details")

    with st.container():
        st.markdown("<div class='card'><div class='card-title'>Vehicle Details</div>", unsafe_allow_html=True)
        year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2015, step=1, format="%d")
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=500, format="%d")
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'><div class='card-title'>Listing & Ownership</div>", unsafe_allow_html=True)
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
        st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("🔍 Predict Price")

if submit:
    input_data = pd.DataFrame(
        [[year, km_driven, fuel, seller_type, transmission, owner]],
        columns=["Year", "KM_Driven", "Fuel", "Seller_Type", "Transmission", "Owner"]
    )

    prediction = model.predict(input_data)[0]
    st.markdown(f'<div class="result-box">Predicted Price: ₹ {prediction:,.2f}</div>', unsafe_allow_html=True)
