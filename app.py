# app.py
# Streamlit app for car price prediction with "name" (car name) as a dropdown
# Works with an sklearn Pipeline (e.g., ColumnTransformer + OneHotEncoder + model)
# Place this file alongside: best_model_GradientBoosting.pickle

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(pickle_path: str):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _flatten(l):
    return [item for sub in l for item in (sub if isinstance(sub, (list, tuple)) else [sub])]


def try_get_preprocessor_and_model(pipeline: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Try to split a composite sklearn Pipeline into (preprocessor, estimator).
    Returns (preprocessor, final_estimator) or (None, pipeline) if unknown.
    """
    # Common patterns:
    # 1) Pipeline(steps=[("preprocessor", coltx), ("model", estimator)])
    # 2) Direct estimator (no pipeline)
    try:
        if hasattr(pipeline, "named_steps"):
            pre = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get("prep")
            est = pipeline.named_steps.get("model") or pipeline.named_steps.get("regressor") or pipeline.steps[-1][1]
            return pre, est
        # Some people wrap ColumnTransformer directly; then the final model may be inside another object.
        return None, pipeline
    except Exception:
        return None, pipeline


def find_input_columns(preprocessor: Any) -> List[str]:
    """
    Best-effort extraction of original input column names from a ColumnTransformer.
    """
    cols = []
    if preprocessor is None:
        return cols

    try:
        # ColumnTransformer has .transformers or .transformers_
        transformers = getattr(preprocessor, "transformers_", None) or getattr(preprocessor, "transformers", [])
        for name, transformer, columns in transformers:
            # columns can be a list of names, a single name, or a callable
            if callable(columns):
                continue
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                cols.extend([c for c in columns if isinstance(c, str)])
            elif isinstance(columns, str):
                cols.append(columns)
    except Exception:
        pass

    # unique and keep order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def get_onehot_categories_for_column(preprocessor: Any, column_name: str) -> List[str]:
    """
    Attempts to locate a OneHotEncoder in the ColumnTransformer and return categories
    for the given original input column (before encoding).
    """
    if preprocessor is None:
        return []

    try:
        transformers = getattr(preprocessor, "transformers_", None) or getattr(preprocessor, "transformers", [])
        for name, transformer, columns in transformers:
            # Some transformers can be Pipelines themselves
            inner = transformer
            if hasattr(transformer, "named_steps"):
                # e.g., Pipeline([("imputer", ...), ("ohe", OneHotEncoder(...))])
                # We want the OneHotEncoder inside if present
                for step_name, step in transformer.named_steps.items():
                    if step.__class__.__name__.lower().startswith("onehotencoder"):
                        ohe = step
                        # map categories back to the columns list
                        cols_list = columns if isinstance(columns, (list, tuple, pd.Index, np.ndarray)) else [columns]
                        if column_name in cols_list:
                            idx = list(cols_list).index(column_name)
                            cats = ohe.categories_[idx] if idx < len(ohe.categories_) else []
                            return list(cats)
                inner = list(transformer.named_steps.values())[-1]  # fallthrough: last step

            # If the transformer itself is an OneHotEncoder
            if inner.__class__.__name__.lower().startswith("onehotencoder"):
                ohe = inner
                cols_list = columns if isinstance(columns, (list, tuple, pd.Index, np.ndarray)) else [columns]
                if column_name in cols_list:
                    idx = list(cols_list).index(column_name)
                    cats = ohe.categories_[idx] if idx < len(ohe.categories_) else []
                    return list(cats)
    except Exception:
        pass

    return []


def default_feature_schema(existing_cols: List[str]) -> List[str]:
    """
    If we couldn't detect full input schema, fall back to a common car-price dataset schema.
    Preserve any detected column order and append missing known columns at the end.
    """
    common = [
        "name", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine", "max_power", "seats"
    ]
    merged = []
    seen = set()
    # keep detected order first
    for c in existing_cols:
        if isinstance(c, str) and c not in seen:
            merged.append(c); seen.add(c)
    # append common ones not present
    for c in common:
        if c not in seen:
            merged.append(c); seen.add(c)
    return merged


# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "best_model_GradientBoosting.pickle"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please place your pickle next to this app.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "Couldn't load the model pickle. Ensure you run this with the same major versions of "
        "NumPy / scikit-learn used to train the model.\n\n"
        f"Loader error:\n{e}"
    )
    st.stop()

preprocessor, final_estimator = try_get_preprocessor_and_model(model)
input_cols = find_input_columns(preprocessor)
required_cols = default_feature_schema(input_cols)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("If your model doesn't expose car names, paste them below to populate the dropdown.")
    car_names_csv = st.text_area(
        "Car names (comma-separated)",
        value="",
        placeholder="Alto 800, Baleno, Swift, i20, Creta, City, ...",
    )
    user_car_names = [x.strip() for x in car_names_csv.split(",") if x.strip()]

    st.markdown("---")
    st.caption("Optional: predict from a CSV (one row per car). "
               "CSV must include the same input columns used by your model.")
    uploaded_csv = st.file_uploader("Upload CSV for batch prediction", type=["csv"], accept_multiple_files=False)

# Try to extract categories for 'name' from the fitted encoder (if present)
model_car_name_options = get_onehot_categories_for_column(preprocessor, "name")

# -----------------------------
# Page Header
# -----------------------------
st.title("🚗 Car Price Predictor")
st.write(
    "Select the **car name** from a dropdown and fill the rest of the features. "
    "This app will use your trained model to predict the price."
)

# -----------------------------
# Single Prediction Form
# -----------------------------
st.subheader("Single Prediction")

form_values = {}
with st.form("single_inference"):
    for col in required_cols:
        # Simple heuristic to choose widget types
        cl = col.lower()

        # --- Special handling for 'name' (car name) ---
        if cl == "name":
            options = model_car_name_options or user_car_names
            if options:
                form_values[col] = st.selectbox("name", options, index=0)
            else:
                form_values[col] = st.text_input("name", value="")
            continue

        # Known categoricals
        if cl in ["fuel", "seller_type", "transmission", "owner"]:
            if cl == "fuel":
                form_values[col] = st.selectbox("fuel", ["petrol", "diesel", "cng", "lpg", "electric"])
            elif cl == "seller_type":
                form_values[col] = st.selectbox("seller_type", ["individual", "dealer", "trustmark_dealer"])
            elif cl == "transmission":
                form_values[col] = st.selectbox("transmission", ["manual", "automatic"])
            elif cl == "owner":
                form_values[col] = st.selectbox("owner", ["first", "second", "third", "fourth & above"])
            continue

        # Numeric-ish columns (int)
        if cl in ["year", "km_driven", "seats"]:
            step = 1
            min_val = 0 if cl != "year" else 1990
            default_val = 2015 if cl == "year" else 0
            form_values[col] = st.number_input(col, min_value=min_val, step=step, value=default_val, format="%d")
            continue

        # Numeric-ish columns (float)
        if cl in ["mileage", "engine", "max_power"]:
            # Allow floats; defaults reasonable
            defaults = {"mileage": 18.0, "engine": 1197.0, "max_power": 82.0}
            form_values[col] = st.number_input(col, value=float(defaults.get(cl, 0.0)))
            continue

        # Fallbacks: try numeric, else text
        # We'll guess numeric if model categories include this column elsewhere (they won't) => use text by default
        # To be safe, use number for unknowns that look numeric by name:
        if any(tok in cl for tok in ["price", "power", "mileage", "engine", "bhp", "torque", "cc"]):
            form_values[col] = st.number_input(col, value=0.0)
        else:
            form_values[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        row_df = pd.DataFrame([form_values])  # single row
        pred = model.predict(row_df)[0]
        st.success(f"💰 Predicted Price: **{pred:,.2f}**")
        with st.expander("Show input row"):
            st.dataframe(row_df)
    except Exception as e:
        st.error("Prediction failed. Please ensure your inputs match the model's training schema.")
        st.exception(e)

# -----------------------------
# Batch Prediction
# -----------------------------
st.subheader("Batch Prediction (CSV)")
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head(20))

        # If 'name' is present and user provided a custom list, optionally validate entries
        if "name" in df.columns and (model_car_name_options or user_car_names):
            valid_names = set(model_car_name_options or user_car_names)
            invalid = df.loc[~df["name"].astype(str).isin(valid_names), "name"].unique().tolist()
            if invalid:
                st.warning(
                    "Some 'name' values are not in the dropdown list. "
                    "The model may still handle them if it was trained that way.\n\n"
                    f"Out-of-list examples: {invalid[:10]}"
                )

        preds = model.predict(df)
        out = df.copy()
        out["predicted_price"] = preds
        st.success("Batch prediction complete.")
        st.dataframe(out.head(50))

        # Offer download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error("Batch prediction failed. Ensure the CSV columns match your model's training schema.")
        st.exception(e)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Tip: If the **name** dropdown is empty, paste your list of car names in the sidebar. "
    "If your model's preprocessor exposes categories for 'name', they'll appear automatically."
)
