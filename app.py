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

# ---- Numpy unpickling shim for some environments ----
try:
    import numpy as _np
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
except Exception:
    pass

# ---- Page setup ----
st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Prediction")

MODEL_PATH = "best_model_GradientBoosting.pickle"

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

# ---------- Helpers to introspect expected columns from the pipeline ----------
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
    # find ColumnTransformer in pipeline (possibly nested)
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

# ---------- Build car-name dropdown options ----------
def detect_car_name_feature(cat_map: Dict[str, List[str]]) -> str:
    # try common names used in datasets/models
    candidates = ["car_name", "name", "car", "model", "car_model", "brand_model"]
    lower_keys = {k.lower(): k for k in cat_map.keys()}
    for cand in candidates:
        if cand in lower_keys:
            return lower_keys[cand]
    return ""

# Preferred: categories from model; Fallback: curated list
car_name_feature = detect_car_name_feature(cat_map)
if car_name_feature:
    car_options = sorted([c for c in cat_map.get(car_name_feature, []) if c and str(c_]()_
