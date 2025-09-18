# app_modern.py
import io
import pickle
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ---------- Utilities to make unpickling more robust ----------
try:
    import numpy as _np
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
except Exception:
    pass

# ---------- Page + minimal theme ----------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# light “card” look + subtle typography polish
st.markdown("""
<style>
:root {
  --card-bg: #ffffff10;
}
.stApp {
  background: radial-gradient(1200px 560px at 20% -10%, #eef3ff 0%, #ffffff 45%) no-repeat,
              linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
}
h1,h2,h3 { letter-spacing: .2px; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
.card {
  border-radius: 16px;
  padding: 20px 20px 10px 20px;
  border: 1px solid #e9eef5;
  background: #fff;
  box-shadow: 0 4px 16px rgba(15,23,42,.06);
}
.card-tight {
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid #edf1f6;
  background: #fff;
  box-shadow: 0 2px 10px rgba(15,23,42,.04);
}
.small-note { color:#6b7280; font-size: 0.9rem; }
.big-cta button { font-size: 1.05rem; height: 3rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #edf1f6; border-radius:12px; padding:14px; }
[data-baseweb="tab-list"] { margin-bottom: .5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>🚗 Used Car Price Prediction</h1>"
    "<div class='small-note'>Professional UI shell over your existing features—no logic changes.</div>",
    unsafe_allow_html=True
)

# ---------- Load model ----------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

MODEL_PATH = "best_model_GradientBoosting.pickle"
with st.spinner("Loading model…"):
    model = load_model(MODEL_PATH)

# ---------- Introspection helpers ----------
def _flatten_transformers(ct: ColumnTransformer) -> List[Tuple[str, Any, List[str]]]:
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
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
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
    for name, transformer, cols in _flatten_transformers(ct):
        enc = transformer
        if isinstance(transformer, Pipeline) and len(transformer.steps) > 0:
            enc = transformer.steps[0][1]
        if isinstance(enc, OneHotEncoder):
            if hasattr(enc, "categories_"):
                for c, col in zip(enc.categories_, cols):
                    result[col] = [str(x) for x in list(c)]
    return result

def get_expected_input_schema(model) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    if isinstance(model, Pipeline):
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
                raise ValueError(
                    "Could not infer required input columns. Please provide a CSV with the original training column names."
                )
        return required_cols, type_map, cat_map
    raise ValueError("Expected an sklearn Pipeline object.")

def coerce_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, t in type_map.items():
        if col not in df.columns:
            continue
        if t == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    return df

# ---------- Schema ----------
try:
    required_cols, type_map, cat_map = get_expected_input_schema(model)
except Exception as e:
    st.error(f"Could not introspect model input schema: {e}")
    st.stop()

# ---------- Top summary strip ----------
colA, colB, colC = st.columns(3)
with colA: st.metric("Model file", MODEL_PATH)
with colB: st.metric("Expected columns", len(required_cols))
with colC: st.metric("Categorical fields", sum(1 for c in required_cols if type_map.get(c)=="categorical"))

st.divider()

# ---------- Tabs: Single vs Batch ----------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch (CSV)", "Schema"])

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(pd.DataFrame({
        "column": required_cols,
        "type (best-effort)": [type_map.get(c, "unknown") for c in required_cols],
        "known categories (if any)": [", ".join(cat_map.get(c, [])) for c in required_cols],
    }))
    st.caption("Use the exact column names & compatible types.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Enter values")
        with st.form("single_inference"):
            form_values = {}
            for col in required_cols:
                inferred_type = type_map.get(col, "numeric")
                label = col.replace("_"," ").title()
                if inferred_type == "categorical":
                    if col in cat_map and len(cat_map[col]) > 0:
                        form_values[col] = st.selectbox(label, cat_map[col], index=0)
                    else:
                        if col.lower() == "fuel":
                            form_values[col] = st.selectbox(label, ["petrol", "diesel", "cng", "lpg", "electric"])
                        elif col.lower() == "seller_type":
                            form_values[col] = st.selectbox(label, ["individual", "dealer", "trustmark_dealer"])
                        elif col.lower() == "transmission":
                            form_values[col] = st.selectbox(label, ["manual", "automatic"])
                        elif col.lower() == "owner":
                            form_values[col] = st.selectbox(label, ["first", "second", "third", "fourth & above"])
                        else:
                            form_values[col] = st.text_input(label, value="")
                else:
                    if col in ["km_driven", "car_age"]:
                        form_values[col] = st.number_input(label, min_value=0, step=1, format="%d")
                    else:
                        form_values[col] = st.number_input(label, value=0.0, step=1.0, format="%.6f")

            submitted = st.form_submit_button("Predict", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            input_df = pd.DataFrame([form_values], columns=required_cols)
            input_df = coerce_types(input_df, type_map)
            missing = [c for c in required_cols if c not in input_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if input_df.isna().any().any():
                    st.warning("Some numeric fields were invalid and coerced to NaN; the model/imputer will handle them if configured.")
                try:
                    pred = model.predict(input_df)[0]
                    with right:
                        st.markdown("<div class='card-tight'>", unsafe_allow_html=True)
                        st.subheader("Result")
                        st.success(f"**Predicted price:** {float(pred):,.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    with right:
        st.markdown("<div class='card-tight'>", unsafe_allow_html=True)
        st.subheader("Tips")
        st.write("- Keep names & types exactly as trained.\n- Use the *Schema* tab for reference.\n- Manual vs Automatic & Owner tiers affect price strongly in most datasets.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload CSV and predict")
    uploaded = st.file_uploader("Upload a CSV with exactly the expected columns (case-sensitive column names).", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        missing_cols = [c for c in required_cols if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in required_cols]

        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.info("Tip: Rename columns to match the expected names shown in the Schema tab.")
        else:
            if extra_cols:
                st.warning(f"CSV has extra columns that will be ignored: {extra_cols}")
                df = df[required_cols]

            df = coerce_types(df, type_map)
            try:
                preds = model.predict(df)
                out = df.copy()
                out["prediction"] = preds
                st.success("Predictions complete.")
                st.dataframe(out.head(20))
                buf = io.BytesIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    "Download predictions as CSV",
                    data=buf.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    st.caption("If you see a 'columns are missing' error, ensure column names and types match the Schema.")
    st.markdown("</div>", unsafe_allow_html=True)
