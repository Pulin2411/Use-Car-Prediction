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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # ✅ added

# ---- Numpy unpickling shim for some environments ----
try:
    import numpy as _np
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
except Exception:
    pass

# ---- Page setup ----
st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")
st.markdown("""
<style>
/* Background + type */
.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, #eaf0ff 0%, #ffffff 50%) no-repeat,
    linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}
h1, h2, h3 { letter-spacing: .2px; }

/* Hero */
.hero {
  display:flex; align-items:center; gap:14px;
  padding: 18px 22px; border-radius: 18px;
  background: rgba(255,255,255,.7);
  border: 1px solid #e6ecf5; box-shadow: 0 6px 18px rgba(10,20,40,.06);
  backdrop-filter: blur(6px);
}

/* Cards */
.card {
  border-radius: 16px; padding: 18px 18px 8px 18px;
  background: rgba(255,255,255,.85);
  border: 1px solid #e9eef5; box-shadow: 0 8px 24px rgba(15,23,42,.06);
  backdrop-filter: blur(6px);
}
.card-title { font-weight:600; margin-bottom: 6px; }

/* Sticky result on the right */
.sticky {
  position: sticky; top: 12px;
}

/* CTA Button */
.big-cta button {
  height: 3.2rem; font-size: 1.08rem; font-weight: 600;
  border-radius: 12px; box-shadow: 0 8px 18px rgba(0,0,0,.06);
}
.small-note { color:#6b7280; font-size:.92rem; }

/* Tidy up spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='hero'><span style='font-size:1.6rem'>🚗</span>"
    "<div><div style='font-size:1.4rem; font-weight:700;'>Used Car Price Prediction</div></div></div>",
    unsafe_allow_html=True
)

MODEL_PATH = "best_model_GradientBoosting.pickle"

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

# ---------- Helpers ----------
def _flatten_transformers(ct: ColumnTransformer):
    flat = []
    for name, transformer, cols in ct.transformers:
        if transformer == "drop":
            continue
        if isinstance(cols, (list, tuple, np.ndarray)):
            cols = list(cols)
        elif isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols) if cols is not None else []
        flat.append((name, transformer, cols))
    return flat

def _get_column_types_from_transformers(ct: ColumnTransformer) -> Dict[str, str]:
    col_types: Dict[str, str] = {}
    for _, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and transformer.steps:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder):
            for c in cols:
                col_types[c] = "categorical"
        else:
            for c in cols:
                col_types.setdefault(c, "numeric")
    return col_types

def _get_ohe_categories(ct: ColumnTransformer) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for _, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and transformer.steps:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder) and hasattr(enc, "categories_"):
            for cats, col in zip(enc.categories_, cols):
                result[col] = [str(x) for x in list(cats)]
    return result

def get_expected_input_schema(model) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    if not isinstance(model, Pipeline):
        raise ValueError("Expected an sklearn Pipeline object.")
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
        raise ValueError("No ColumnTransformer found in the pipeline.")

    required_cols: List[str] = []
    for _, _, cols in _flatten_transformers(preproc):
        required_cols.extend(cols)

    type_map = _get_column_types_from_transformers(preproc)
    cat_map = _get_ohe_categories(preproc)

    if not required_cols:
        if hasattr(model, "feature_names_in_"):
            required_cols = list(model.feature_names_in_)
        else:
            raise ValueError("Could not infer required input columns from the model.")

    return required_cols, type_map, cat_map

# Get model schema
try:
    required_cols, type_map, cat_map = get_expected_input_schema(model)
except Exception as e:
    st.error(f"Could not introspect model input schema: {e}")
    st.stop()

# ---------- Car-name dropdown ----------
def detect_car_name_feature(cat_map: Dict[str, List[str]]) -> str:
    candidates = ["car_name", "name", "car", "model", "car_model", "brand_model"]
    lower_keys = {k.lower(): k for k in cat_map.keys()}
    for cand in candidates:
        if cand in lower_keys:
            return lower_keys[cand]
    return ""

car_name_feature = detect_car_name_feature(cat_map)
if car_name_feature:
    car_options = sorted([c for c in cat_map.get(car_name_feature, []) if c and str(c).strip() != ""])
else:
    car_options = [
        "Maruti Swift", "Maruti Alto", "Maruti Baleno", "Maruti Dzire",
        "Hyundai i10", "Hyundai i20", "Hyundai Creta", "Hyundai Verna",
        "Honda City", "Honda Amaze",
        "Tata Nexon", "Tata Tiago", "Tata Altroz",
        "Toyota Innova", "Toyota Glanza",
        "Mahindra XUV500", "Mahindra Scorpio",
        "Ford EcoSport",
        "Renault Kwid", "Skoda Rapid", "Volkswagen Polo"
    ]

# ---------- Layout ----------
left, right = st.columns([1.25, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Vehicle Details</div>", unsafe_allow_html=True)
    car_name = st.selectbox("Car Name (for display)", car_options, index=car_options.index("Maruti Swift") if "Maruti Swift" in car_options else 0)
    year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2015, step=1, format="%d")
    km_driven = st.number_input("KM Driven", min_value=0, value=50000, step=500, format="%d")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Listing & Ownership</div>", unsafe_allow_html=True)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='big-cta' style='margin-top:14px;'>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Selling Price", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card sticky'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Prediction</div>", unsafe_allow_html=True)
    result_container = st.empty()
    st.caption("Tip: Car Name is a dropdown. If your model includes a car-name feature, options are loaded from it automatically.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Build input row ----------
def normalize(s: str) -> str:
    return str(s).strip().lower()

def build_input_row() -> pd.DataFrame:
    row = {c: np.nan for c in required_cols}
    cols_lower = {c.lower(): c for c in required_cols}

    for candidate in ["car_name", "name", "car", "model", "car_model", "brand_model"]:
        if candidate in cols_lower:
            row[cols_lower[candidate]] = car_name
            break

    if "year" in cols_lower:
        row[cols_lower["year"]] = int(year)
    elif "car_age" in cols_lower:
        row[cols_lower["car_age"]] = int(max(0, datetime.now().year - int(year)))

    for candidate in ["km_driven", "kms_driven", "kilometers_driven"]:
        if candidate in cols_lower:
            row[cols_lower[candidate]] = int(km_driven)
            break

    fuel_val = normalize(fuel)
    fuel_map = {"petrol": "petrol", "diesel": "diesel", "cng": "cng", "lpg": "lpg", "electric": "electric"}
    if "fuel" in cols_lower:
        row[cols_lower["fuel"]] = fuel_map.get(fuel_val, fuel_val)

    seller_val = normalize(seller_type)
    seller_map = {"individual": "individual", "dealer": "dealer", "trustmark dealer": "trustmark_dealer"}
    if "seller_type" in cols_lower:
        row[cols_lower["seller_type"]] = seller_map.get(seller_val, seller_val)

    trans_val = normalize(transmission)
    trans_map = {"manual": "manual", "automatic": "automatic"}
    if "transmission" in cols_lower:
        row[cols_lower["transmission"]] = trans_map.get(trans_val, trans_val)

    owner_val = normalize(owner_type)
    owner_map = {
        "first owner": "first",
        "second owner": "second",
        "third owner": "third",
        "fourth & above": "fourth & above",
    }
    for candidate in ["owner", "owner_type"]:
        if candidate in cols_lower:
            row[cols_lower[candidate]] = owner_map.get(owner_val, owner_val)
            break

    for col, kind in type_map.items():
        if col not in row:
            continue
        if kind == "numeric":
            try:
                row[col] = pd.to_numeric(row[col], errors="coerce")
            except Exception:
                row[col] = np.nan
        else:
            if pd.isna(row[col]):
                row[col] = ""
            row[col] = str(row[col])

    return pd.DataFrame([row], columns=required_cols)

# ---------- Predict ----------
if predict_clicked:
    input_df = build_input_row()
    try:
        pred = model.predict(input_df)[0]
        result_container.success(f"**Predicted price:** ₹{float(pred):,.0f}")
    except Exception as e:
        result_container.error(f"Prediction failed: {e}")

# ---------- Evaluation Section ----------
st.markdown("---")
st.header("📊 Model Performance Metrics")

uploaded_file = st.file_uploader("Upload a CSV file with actual and predicted prices", type=["csv"])

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

        st.subheader("🔎 Sample Data Preview")
        st.dataframe(df.head())
    else:
        st.error("CSV must contain 'actual' and 'predicted' columns.")
