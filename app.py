#!/usr/bin/env python3
import pickle
import pandas as pd
import streamlit as st

# ---------------- Config ----------------
MODEL_PATH = "best_model_GradientBoosting.pickle"
CURRENT_YEAR = 2025

# ---------------- Load Model ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.caption(f"Model loaded from: {MODEL_PATH}")

# ---------------- Single Prediction ----------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1980, max_value=CURRENT_YEAR, value=2016, step=1)
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Unknown"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Unknown"])
    with col2:
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=52000, step=100)
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer", "Unknown"])
        owner = st.selectbox("Owner", [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
            "Unknown",
        ])

    submitted = st.form_submit_button("Predict")

if submitted:
    car_age = CURRENT_YEAR - int(year)
    X = pd.DataFrame([{
        "km_driven": float(km_driven),
        "car_age": float(car_age),
        "fuel": str(fuel),
        "seller_type": str(seller_type),
        "transmission": str(transmission),
        "owner": str(owner),
    }])
    try:
        pred = float(model.predict(X)[0])
        st.success(f"Estimated Price: ₹ {pred:,.0f}")
        with st.expander("View input features"):
            st.write(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- Batch Predictions ----------------
st.markdown("---")
st.subheader("Batch predictions (CSV)")
st.caption("Upload a CSV with columns: year, km_driven, fuel, seller_type, transmission, owner.")

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)
        if "year" in df.columns:
            df["car_age"] = CURRENT_YEAR - pd.to_numeric(df["year"], errors="coerce")

        required = ["km_driven", "car_age", "fuel", "seller_type", "transmission", "owner"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns after processing: {missing}")
        else:
            preds = model.predict(df[required])
            out = df.copy()
            out["predicted_price"] = preds
            st.dataframe(out.head(20))
            st.download_button("Download predictions", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
