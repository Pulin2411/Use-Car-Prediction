import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load the Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model_GradientBoosting.joblib")

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Prediction App")
st.write("Estimate the **selling price** of a used car based on its features.")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Car Details")

col1, col2 = st.columns(2)

with col1:
    car_age = st.slider("Car Age (years)", min_value=1, max_value=30, value=5)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Convert categorical inputs to numerical (example mapping – adjust based on your training data)
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
trans_mapping = {"Manual": 0, "Automatic": 1}

fuel_type_encoded = fuel_mapping[fuel_type]
transmission_encoded = trans_mapping[transmission]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Price"):
    try:
        input_data = np.array([[car_age, kms_driven, fuel_type_encoded, transmission_encoded]])
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------
# Footer / Info
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **How it works:**  
    This app uses a Gradient Boosting Regression model trained on historical car sales data.  
    It predicts the selling price based on car age, kilometers driven, fuel type, and transmission.
    """
)
