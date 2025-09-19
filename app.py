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
st.set_page_config(
    page_title="🚗 Used Car Predictor By Pulin",
    page_icon="🚗",
    layout="wide"
)

# ---- App Title (sticky + shrinkable header with emoji logo) ----
st.markdown(
    """
    <div class="app-header" id="appHeader">
        <span class="logo">🚗</span>
        <span class="title">Used Car Predictor By Pulin</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Styling ----
st.markdown("""
<style>
.stApp {
  background: linear-gradient(180deg, #f9fbfd 0%, #eef2f7 100%);
  color: #1f2937;
  scroll-behavior: smooth;
}

/* Sticky Header (blue bar with emoji logo) */
.app-header {
  position: sticky;
  top: 0;
  width: 100%;
  background: rgba(37, 99, 235, 0.95); /* semi-transparent for blur */
  display: flex;
  align-items: center;
  gap: 12px;
  justify-content: center;
  font-size: 26px;
  font-weight: 700;
  padding: 14px;
  z-index: 1000;
  color: #ffffff;
  box-shadow: 0 2px 8px rgba(0,0,0,.2);
  transition: all 0.28s ease;
  will-change: padding, font-size, transform, box-shadow, opacity, backdrop-filter;
  backdrop-filter: blur(0px);
}
.app-header .logo {
  font-size: 28px;
  transition: all 0.28s ease;
}
.app-header .title {
  font-size: 26px;
  font-weight: 700;
  transition: all 0.28s ease;
}

/* Fade-in animation used when shrinking */
@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Shrinked Header (with blur + subtle fade/slide) */
.app-header.shrink {
  padding: 6px 12px;
  font-size: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,.3);
  opacity: 0.96;
  animation: fadeInDown 0.25s ease;
  backdrop-filter: blur(6px); /* Blur applied when shrinking */
}
.app-header.shrink .logo {
  font-size: 22px;
}
.app-header.shrink .title {
  font-size: 20px;
}

/* Tabs styling */
.stTabs [role="tab"] {
  background: #fff !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  margin-right: 6px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(0,0,0,.05) !important;
  opacity: 1 !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  border: 2px solid #2563eb !important;
}

/* Cards */
.card {
  border-radius: 12px;
  padding: 18px;
  background: #ffffff;
  border: 1px solid #e5eaf0;
  box-shadow: 0 4px 12px rgba(0,0,0,.05);
  margin-bottom: 16px;
}
.card-title { font-weight:600; margin-bottom: 10px; font-size: 1.1rem; }

/* Buttons */
.stButton>button {
  width: 100% !important;
  border-radius: 12px !important;
  padding: 12px !important;
  font-size: 1rem !important;
  background-color: #2563eb !important;
  color: white !important;
  border: none !important;
  opacity: 1 !important;
}
.stButton>button:hover {
  background-color: #1e40af !important;
}

/* Result */
.result-box {
  padding: 18px;
  border-radius: 12px;
  background: #f0f7ff;
  text-align: center;
  font-size: 1.3rem;
  font-weight: 600;
  margin-top: 10px;
  border: 1px solid #dbeafe;
}

/* Sticky bottom bar for mobile-first style */
.sticky-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  background: #2563eb;
  padding: 12px;
  text-align: center;
  z-index: 999;
  box-shadow: 0 -4px 8px rgba(0,0,0,.1);
}
.sticky-bar button {
  width: 90% !important;
  background: #ffffff !important;
  color: #2563eb !important;
  font-weight: bold !important;
  font-size: 1rem !important;
  border-radius: 10px !important;
  padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---- JavaScript for Shrink Effect ----
st.markdown("""
<script>
(function() {
  const header = document.getElementById('appHeader');
  if (!header) return;

  const THRESHOLD = 50;
  let shrunk = false;

  const onScroll = () => {
    const shouldShrink = window.scrollY > THRESHOLD;
    if (shouldShrink !== shrunk) {
      shrunk = shouldShrink;
      if (shrunk) {
        header.classList.add('shrink');
      } else {
        header.classList.remove('shrink');
      }
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
</script>
""", unsafe_allow_html=True)

# ---- Load Model ----
MODEL_PATH = "best_model_GradientBoosting.pickle"
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(MODEL_PATH)

# ---- Extract car options dynamically ----
def get_car_options(model):
    if not isinstance(model, Pipeline):
        return []
    preproc = None
    for _, step in model.steps:
        if isinstance(step, ColumnTransformer):
            preproc = step
            break
        if isinstance(step, Pipeline):
            for _, inner in step.steps:
                if isinstance(inner, ColumnTransformer):
                    preproc = inner
                    break
    if preproc is None:
        return []

    for _, transformer, cols in preproc.transformers:
        enc = transformer
        if isinstance(transformer, Pipeline) and transformer.steps:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder) and hasattr(enc, "categories_"):
            for cats, col in zip(enc.categories_, cols):
                if str(col).lower() in ["car_name", "name", "model", "car_model", "brand_model"]:
                    return sorted([str(x) for x in cats if str(x).strip() != ""])
    return []

car_options = get_car_options(model)
if not car_options:
    car_options = [
        "Maruti Swift", "Maruti Alto", "Maruti Baleno", "Maruti Dzire",
        "Hyundai i10", "Hyundai i20", "Hyundai Creta", "Hyundai Verna",
        "Honda City", "Honda Amaze",
        "Tata Nexon", "Tata Tiago", "Tata Altroz",
        "Toyota Innova", "Toyota Glanza",
        "Mahindra XUV500", "Mahindra Scorpio",
        "Ford EcoSport", "Renault Kwid", "Skoda Rapid", "Volkswagen Polo"
    ]

# ---- Tabs ----
tabs = st.tabs(["Inputs", "Prediction", "Metrics"])

# ---- Tab 1: Inputs ----
with tabs[0]:
    st.markdown("<div class='card'><div class='card-title'>Enter Car Details</div>", unsafe_allow_html=True)
    car_name = st.selectbox("Car Name", car_options, index=0)
    year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2015, step=1, format="%d")
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=500, format="%d")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
    predict_clicked = st.button("🔍 Predict Price")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Prediction logic ----
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if predict_clicked:
    car_age = datetime.now().year - year
    input_data = pd.DataFrame(
        [[car_name, car_age, km_driven, fuel, seller_type, transmission, owner]],
        columns=["car_name", "car_age", "km_driven", "fuel", "seller_type", "transmission", "owner"]
    )
    try:
        st.session_state.prediction = float(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---- Tab 2: Prediction ----
with tabs[1]:
    if st.session_state.prediction is None:
        st.info("Go to Inputs tab and click Predict.")
    else:
        price = float(st.session_state.prediction)
        st.markdown(
            f'<div class="result-box">Predicted Price: ₹ {price:,.2f}</div>',
            unsafe_allow_html=True
        )

# ---- Tab 3: Metrics ----
with tabs[2]:
    st.subheader("📊 Model Performance Metrics")
    uploaded_file = st.file_uploader("Upload a CSV with actual and predicted prices", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "actual" in df.columns and "predicted" in df.columns:
            y_test = df["actual"]
            y_pred = df["predicted"]
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**MAE:** ₹{mae:,.0f}")
            st.write(f"**RMSE:** ₹{rmse:,.0f}")
            st.dataframe(df.head())
        else:
            st.error("CSV must contain 'actual' and 'predicted' columns.")

# ---- Sticky bottom bar (mobile-first style) ----
if st.session_state.prediction is not None:
    try:
        price = float(st.session_state.prediction)
        st.markdown(
            f"""
            <div class='sticky-bar'>
                <button disabled>Predicted Price: ₹ {price:,.2f}</button>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction could not be displayed in sticky bar: {e}")
